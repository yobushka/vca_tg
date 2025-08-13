import os
import time
import threading
import requests
import json
import yaml
from datetime import datetime, timezone
from onvif import ONVIFCamera
from zeep.exceptions import Fault
from zeep.helpers import serialize_object
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
import re
from io import BytesIO
import itertools
from pathlib import Path
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
try:
    # zeep depends on lxml; use it, but keep optional at import time
    from lxml import etree as ET
except Exception:  # pragma: no cover
    ET = None
from urllib.parse import urlparse, urlunparse

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
CLIP_SECONDS = int(os.environ.get("CLIP_SECONDS", "10"))
COOLDOWN_SECONDS = int(os.environ.get("COOLDOWN_SECONDS", "20"))
DEBUG = int(os.environ.get("DEBUG", "1"))
CAP_TEST = int(os.environ.get("CAP_TEST", "1"))
RAW_XML_SAVE = int(os.environ.get("RAW_XML_SAVE", "0"))  # 1 -> also save each raw XML to files
RAW_XML_DIR = os.environ.get("RAW_XML_DIR", "raw_xml")
RAW_XML_MAX_PRINT = int(os.environ.get("RAW_XML_MAX_PRINT", "4000"))  # truncate console output
DUMP_EVENT_PROPERTIES = int(os.environ.get("DUMP_EVENT_PROPERTIES", "1"))  # dump GetEventProperties once per camera

_raw_xml_counter = itertools.count(1)

if not BOT_TOKEN or not CHAT_ID:
    raise SystemExit("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.")

CONFIG_PATH = os.environ.get("CONFIG", "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

TOPICS_FILTER = [s.lower() for s in cfg.get("topics_filter", [])]
PREFER_STREAM = int(cfg.get("prefer_stream", 1))
IGNORE_FILTER = [s.lower() for s in cfg.get("ignore_filter", [])]

def vlog(msg):
    if DEBUG:
        try:
            print(msg)
        except Exception:
            # Be resilient to non-utf8 consoles
            print(str(msg).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))


def _extract_raw_xml_element(m):
    """Return raw XML unicode string for m.Message._value_1 if available."""
    try:
        msg = getattr(m, 'Message', None)
        if msg is not None and hasattr(msg, '_value_1'):
            elem = getattr(msg, '_value_1')
            if ET is not None and hasattr(elem, 'tag'):
                try:
                    return ET.tostring(elem, encoding='unicode')
                except Exception:
                    # Fallback: bytes decode
                    try:
                        return ET.tostring(elem, encoding='utf-8').decode('utf-8', 'ignore')
                    except Exception:
                        return None
    except Exception:
        return None
    return None


def _maybe_dump_raw_xml(m, name_tag, stage):
    """Dump raw XML (print + optional save) once per stage per message."""
    if not DEBUG:
        return
    try:
        dumped = getattr(m, '_raw_xml_dumped_stages', None)
        if dumped is None:
            dumped = set()
            try:
                setattr(m, '_raw_xml_dumped_stages', dumped)
            except Exception:
                pass
        if stage in dumped:
            return
        xml_str = _extract_raw_xml_element(m)
        if not xml_str:
            return
        dumped.add(stage)
        # Print (truncate for readability)
        header = f"[{name_tag}] RAW XML ({stage}) length={len(xml_str)}"
        print("\n" + header)
        print("-" * len(header))
        if len(xml_str) > RAW_XML_MAX_PRINT:
            print(xml_str[:RAW_XML_MAX_PRINT] + f"\n...<truncated {len(xml_str)-RAW_XML_MAX_PRINT} chars>...")
        else:
            print(xml_str)
        # Optional file save
        if RAW_XML_SAVE:
            try:
                Path(RAW_XML_DIR).mkdir(parents=True, exist_ok=True)
                ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
                idx = next(_raw_xml_counter)
                safe_cam = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name_tag))[:50]
                fname = f"{ts}_{idx:06d}_{safe_cam}_{stage}.xml"
                fpath = Path(RAW_XML_DIR) / fname
                with open(fpath, 'w', encoding='utf-8') as fh:
                    fh.write(xml_str)
            except Exception as e:
                print(f"[{name_tag}] RAW XML save failed: {e}")
    except Exception as e:
        print(f"[{name_tag}] RAW XML dump error: {e}")

def debug_dump_event(m, name_tag):
    """Dump full event structure for debugging human detection"""
    if not DEBUG:
        return
    
    print(f"\n{'='*60}")
    print(f"[{name_tag}] FULL EVENT DUMP:")
    print(f"{'='*60}")
    
    try:
        # Try to serialize the full message
        obj = serialize_object(m)
        print(f"[{name_tag}] Serialized object:")
        print(json.dumps(obj, indent=2, default=str))
    except Exception as e:
        print(f"[{name_tag}] Failed to serialize object: {e}")

    # Raw XML dump (stage 'debug_dump_event')
    _maybe_dump_raw_xml(m, name_tag, 'debug_dump_event')
        
    # Try to access raw attributes
    print(f"\n[{name_tag}] Raw message attributes:")
    try:
        for attr in dir(m):
            if not attr.startswith('_'):
                try:
                    val = getattr(m, attr)
                    print(f"  {attr}: {type(val)} = {str(val)[:200]}")
                except Exception as e:
                    print(f"  {attr}: <error accessing: {e}>")
    except Exception as e:
        print(f"  Error listing attributes: {e}")
        
    # Try to access Message specifically
    print(f"\n[{name_tag}] Message structure:")
    try:
        msg = getattr(m, 'Message', None)
        if msg:
            print(f"  Message type: {type(msg)}")
            for attr in dir(msg):
                if not attr.startswith('_'):
                    try:
                        val = getattr(msg, attr)
                        print(f"    Message.{attr}: {type(val)} = {str(val)[:200]}")
                    except Exception as e:
                        print(f"    Message.{attr}: <error: {e}>")
    except Exception as e:
        print(f"  Error accessing Message: {e}")
        
    print(f"{'='*60}\n")

def tg_send_photo(caption, img_bytes, filename="snapshot.jpg"):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {"photo": (filename, img_bytes, "image/jpeg")}
    data = {"chat_id": CHAT_ID, "caption": caption}
    r = requests.post(url, data=data, files=files, timeout=30)
    r.raise_for_status()

def tg_send_video(caption, video_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
    with open(video_path, "rb") as f:
        files = {"video": (os.path.basename(video_path), f, "video/mp4")}
        data = {"chat_id": CHAT_ID, "caption": caption}
        r = requests.post(url, data=data, files=files, timeout=600)
        r.raise_for_status()

def sanitize_snapshot_uri(uri, ip):
    """Normalize snapshot URL without embedding credentials.
    - Ensure scheme/netloc; fallback to http://<ip> when missing.
    - Preserve path/query from camera response.
    """
    u = urlparse(uri or "")
    scheme = u.scheme or "http"
    netloc = u.netloc or (f"{ip}:80")
    # Strip any userinfo if present; we will use HTTP auth headers instead
    try:
        if "@" in netloc:
            netloc = netloc.split("@", 1)[-1]
    except Exception:
        pass
    return urlunparse((scheme, netloc, u.path or "/", u.params, u.query, u.fragment))

def get_profiles(media):
    profiles = media.GetProfiles()
    return profiles

def get_stream_uri(media, profile_token):
    req = media.create_type('GetStreamUri')
    req.StreamSetup = {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}
    req.ProfileToken = profile_token
    resp = media.GetStreamUri(req)
    return resp.Uri

def get_snapshot_uri(media, profile_token):
    req = media.create_type('GetSnapshotUri')
    req.ProfileToken = profile_token
    resp = media.GetSnapshotUri(req)
    return resp.Uri

def ffmpeg_record(rtsp_uri, seconds, out_path):
    # Use ffmpeg to record a short clip from RTSP
    # -y overwrite, -t duration, copy codecs to avoid re-encode
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", rtsp_uri,
        "-t", str(seconds),
        "-an",  # no audio (many cams lack audio)
        "-vcodec", "copy",
        out_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_event_text(m):
    """Build a readable string from ONVIF NotificationMessage for filtering.
    Tries, in order:
    - Topic._value_1 when it's a string
    - Parse XML Message element directly
    - Flatten XML elements (SimpleItem Name/Value, tag local names)
    - Fallback to serialized dict traversal
    """

    def local(tag):
        try:
            return tag.split('}')[-1] if tag and '}' in tag else tag
        except Exception:
            return str(tag)

    def flatten_xml(elem, out):
        if elem is None:
            return
        # If lxml is available and element is an lxml node
        if ET is not None and hasattr(elem, 'tag'):
            try:
                out.append(local(elem.tag))
            except Exception:
                pass
            # Attributes as key=value
            try:
                for k, v in getattr(elem, 'attrib', {}).items():
                    out.append(f"{local(k)}={v}")
            except Exception:
                pass
            # Text content
            try:
                if elem.text and elem.text.strip():
                    out.append(elem.text.strip())
            except Exception:
                pass
            # Children
            try:
                for ch in list(elem):
                    flatten_xml(ch, out)
            except Exception:
                pass
            return
        # Non-lxml unknown — last resort string repr
        try:
            s = str(elem)
            if s:
                out.append(s)
        except Exception:
            pass

    def collect_any(x, out):
        if x is None:
            return
        # Base scalars
        if isinstance(x, (str, int, float)):
            out.append(str(x))
            return
        if isinstance(x, bool):
            out.append("true" if x else "false")
            return
        # XML element
        if hasattr(x, 'tag'):
            flatten_xml(x, out)
            return
        # Mapping
        if isinstance(x, dict):
            for k, v in x.items():
                # Keys sometimes carry meaningful words too
                try:
                    if isinstance(k, str):
                        out.append(k)
                except Exception:
                    pass
                collect_any(v, out)
            return
        # Sequence
        if isinstance(x, (list, tuple, set)):
            for it in x:
                collect_any(it, out)
            return
        # Object with interesting attributes
        for attr in ('_value_1', 'Value', 'value', 'Message', 'Data', 'Source', 'Topic', 'Key', 'SimpleItem', 'ElementItem'):
            try:
                if hasattr(x, attr):
                    collect_any(getattr(x, attr), out)
            except Exception:
                pass

    parts = []
    
    # Topic path (often contains RuleEngine/Line/Motion)
    try:
        t = getattr(m, 'Topic', None)
        if t is not None:
            val = getattr(t, '_value_1', None) or getattr(t, 'Value', None) or getattr(t, 'value', None)
            if isinstance(val, str):
                parts.append(val)
            else:
                collect_any(val, parts)
    except Exception:
        pass
    
    # Direct XML Message parsing (new approach)
    try:
        msg = getattr(m, 'Message', None)
        if msg is not None and hasattr(msg, '_value_1') and hasattr(msg._value_1, 'tag'):
            # This is an XML element, parse it directly
            xml_elem = msg._value_1
            if DEBUG:
                print(f"    DEBUG extract_event_text: Processing XML Message element")
            flatten_xml(xml_elem, parts)
    except Exception as e:
        if DEBUG:
            print(f"    DEBUG extract_event_text: XML parsing failed: {e}")
    
    # Message payload (SimpleItem Name/Value are here) - fallback method
    try:
        msg = getattr(m, 'Message', None)
        if msg is not None:
            # Common containers
            for attr in ('Message', 'Data', 'Source', '_value_1'):
                try:
                    collect_any(getattr(msg, attr, None), parts)
                except Exception:
                    pass
    except Exception:
        pass
    
    # Absolute fallback — serialize entire message
    if not parts:
        try:
            obj = serialize_object(m)
            collect_any(obj, parts)
        except Exception:
            pass

    # Compact and unique-ish
    try:
        # Keep order, remove duplicates
        seen = set()
        uniq = []
        for p in parts:
            if not p:
                continue
            if p in seen:
                continue
            seen.add(p)
            uniq.append(p)
        parts = uniq
    except Exception:
        pass

    result = " ".join([p for p in parts if isinstance(p, str) and p])
    
    if DEBUG:
        print(f"    DEBUG extract_event_text result: '{result[:100]}...'")
    
    return result


def _to_boolish(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "on"}


def detect_trigger(m, topic_text, name_tag='unknown'):
    """Best-effort detection of the fired trigger (e.g., Motion, Tripwire).
    Looks into Message.Data/Source SimpleItem Name/Value pairs and topic text.
    Returns a short label like 'Motion', 'Intrusion', 'CrossLine', 'Human', etc.
    """
    # Raw XML dump early (stage 'detect_trigger')
    _maybe_dump_raw_xml(m, name_tag, 'detect_trigger')
    # Use our improved serialization
    obj = _serialize_message(m)
    
    simple_items = []

    def collect_simple_items(x):
        if x is None:
            return
        if isinstance(x, dict):
            # Common ONVIF shapes: {'SimpleItem': [{'Name': 'IsMotion','Value': 'true'}, ...]}
            if 'SimpleItem' in x:
                si = x['SimpleItem']
                if isinstance(si, list):
                    for it in si:
                        if isinstance(it, dict):
                            simple_items.append((str(it.get('Name', '')), it.get('Value')))
                elif isinstance(si, dict):
                    simple_items.append((str(si.get('Name', '')), si.get('Value')))
            # Recurse into nested
            for v in x.values():
                collect_simple_items(v)
        elif isinstance(x, (list, tuple)):
            for it in x:
                collect_simple_items(it)

    collect_simple_items(obj)

    # Debug output for trigger detection
    if DEBUG:
        print(f"    DEBUG detect_trigger: Found {len(simple_items)} SimpleItems for trigger detection")
        for name, value in simple_items:
            print(f"      Trigger SimpleItem: Name='{name}' Value='{value}'")

    # Heuristics based on SimpleItem Name/Value
    name_val = [(n or "", v) for (n, v) in simple_items]
    for n, v in name_val:
        ln = n.lower()
        if _to_boolish(v) or (v is None and any(k in ln for k in ('motion','trip','cross','intrusion','tamper','audio','human','person','vehicle'))):
            # Motion
            if 'motion' in ln or ln in {'ismotion','motion'}:
                if DEBUG:
                    print(f"    DEBUG: Motion trigger detected via '{n}'='{v}'")
                return 'Motion'
            # Intrusion
            if 'intrusion' in ln:
                if DEBUG:
                    print(f"    DEBUG: Intrusion trigger detected via '{n}'='{v}'")
                return 'Intrusion'
            # Trip/Cross/Line
            if 'trip' in ln and 'line' in ln:
                if DEBUG:
                    print(f"    DEBUG: Tripwire trigger detected via '{n}'='{v}'")
                return 'Tripwire'
            if 'cross' in ln and 'line' in ln:
                if DEBUG:
                    print(f"    DEBUG: CrossLine trigger detected via '{n}'='{v}'")
                return 'CrossLine'
            if 'line' in ln:
                if DEBUG:
                    print(f"    DEBUG: Line trigger detected via '{n}'='{v}'")
                return 'Line'
            if 'cross' in ln:
                if DEBUG:
                    print(f"    DEBUG: Cross trigger detected via '{n}'='{v}'")
                return 'Cross'
            if 'trip' in ln:
                if DEBUG:
                    print(f"    DEBUG: Trip trigger detected via '{n}'='{v}'")
                return 'Trip'
            # Human/Person
            if 'human' in ln or 'person' in ln or 'people' in ln:
                if DEBUG:
                    print(f"    DEBUG: Human trigger detected via '{n}'='{v}'")
                return 'Human'
            # Vehicle
            if 'vehicle' in ln or 'car' in ln or 'truck' in ln or 'bus' in ln or 'bike' in ln:
                if DEBUG:
                    print(f"    DEBUG: Vehicle trigger detected via '{n}'='{v}'")
                return 'Vehicle'
            # Tamper
            if 'tamper' in ln:
                if DEBUG:
                    print(f"    DEBUG: Tamper trigger detected via '{n}'='{v}'")
                return 'Tamper'
            # Audio
            if 'audio' in ln or 'sound' in ln:
                if DEBUG:
                    print(f"    DEBUG: Audio trigger detected via '{n}'='{v}'")
                return 'Audio'

    # Topic path fallback
    tt = (topic_text or '').lower()
    if DEBUG:
        print(f"    DEBUG: Checking topic text for trigger: '{tt[:100]}...'")
    
    m = re.search(r"\b(motion|intrusion|tripwire|crossline|cross|line|human|person|vehicle|tamper|audio)\b", tt)
    if m:
        val = m.group(1)
        # Normalize
        mapping = {
            'cross': 'Cross',
            'line': 'Line',
            'person': 'Human',
        }
        result = mapping.get(val, val.capitalize())
        if DEBUG:
            print(f"    DEBUG: Trigger '{result}' detected via topic text pattern")
        return result

    # Rule name could hint (e.g., MyMotionDetectorRule)
    rm = re.search(r"\b([A-Za-z]*Motion[A-Za-z]*)\b", tt)
    if rm:
        if DEBUG:
            print(f"    DEBUG: Motion trigger detected via rule name pattern")
        return 'Motion'

    if DEBUG:
        print(f"    DEBUG: No specific trigger detected, defaulting to 'Event'")
    return 'Event'


def _parse_xml_element(elem):
    """Parse XML element to extract SimpleItems and other data"""
    if elem is None:
        return {}
    
    result = {}
    
    # Try to process XML element using lxml if available
    if ET is not None and hasattr(elem, 'tag'):
        try:
            # Extract text content
            if elem.text and elem.text.strip():
                result['_text'] = elem.text.strip()
            
            # Extract attributes
            if hasattr(elem, 'attrib') and elem.attrib:
                result['_attributes'] = dict(elem.attrib)
            
            # Process children
            for child in elem:
                tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                
                if tag_name == 'Source':
                    result['Source'] = _parse_source_data(child)
                elif tag_name == 'Data':
                    result['Data'] = _parse_data(child)
                else:
                    # Generic child processing
                    child_data = _parse_xml_element(child)
                    if tag_name in result:
                        if not isinstance(result[tag_name], list):
                            result[tag_name] = [result[tag_name]]
                        result[tag_name].append(child_data)
                    else:
                        result[tag_name] = child_data
                        
        except Exception as e:
            if DEBUG:
                print(f"    DEBUG: XML parsing error: {e}")
            result['_error'] = str(e)
    
    return result

def _parse_source_data(source_elem):
    """Parse Source element to extract SimpleItems"""
    source_data = {}
    simple_items = []
    element_items = []
    
    try:
        for child in source_elem:
            tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            
            if tag_name == 'SimpleItem':
                item = {}
                if hasattr(child, 'attrib'):
                    item.update(child.attrib)
                if child.text and child.text.strip():
                    item['Value'] = child.text.strip()
                simple_items.append(item)
            elif tag_name == 'ElementItem':
                # ElementItem(Name=..., Value=complex subtree with shapes)
                elem_item = {}
                if hasattr(child, 'attrib') and child.attrib:
                    elem_item.update(child.attrib)
                # Parse its children (could include Appearance/Shape/BoundingBox/Polygon)
                parsed_children = []
                for ch2 in child:
                    parsed_children.append(_parse_xml_element(ch2))
                if parsed_children:
                    elem_item['Children'] = parsed_children
                element_items.append(elem_item)
            else:
                if child.text:
                    source_data[tag_name] = child.text.strip()
                elif hasattr(child, 'attrib'):
                    source_data[tag_name] = dict(child.attrib)
    
    except Exception as e:
        if DEBUG:
            print(f"    DEBUG: Source parsing error: {e}")
    
    if simple_items:
        source_data['SimpleItem'] = simple_items
    if element_items:
        source_data['ElementItem'] = element_items
    
    return source_data

def _parse_data(data_elem):
    """Parse Data element to extract SimpleItems"""
    data_data = {}
    simple_items = []
    element_items = []
    
    try:
        for child in data_elem:
            tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            
            if tag_name == 'SimpleItem':
                item = {}
                if hasattr(child, 'attrib'):
                    item.update(child.attrib)
                if child.text and child.text.strip():
                    item['Value'] = child.text.strip()
                simple_items.append(item)
            elif tag_name == 'ElementItem':
                elem_item = {}
                if hasattr(child, 'attrib') and child.attrib:
                    elem_item.update(child.attrib)
                parsed_children = []
                for ch2 in child:
                    parsed_children.append(_parse_xml_element(ch2))
                if parsed_children:
                    elem_item['Children'] = parsed_children
                element_items.append(elem_item)
            else:
                if child.text:
                    data_data[tag_name] = child.text.strip()
                elif hasattr(child, 'attrib'):
                    data_data[tag_name] = dict(child.attrib)
    
    except Exception as e:
        if DEBUG:
            print(f"    DEBUG: Data parsing error: {e}")
    
    if simple_items:
        data_data['SimpleItem'] = simple_items
    if element_items:
        data_data['ElementItem'] = element_items
    
    return data_data


def _serialize_message(m):
    try:
        # First try to get the raw Message XML element
        msg = getattr(m, 'Message', None)
        if msg is not None:
            # Check if it's an XML element that we need to parse
            if hasattr(msg, '_value_1') and hasattr(msg._value_1, 'tag'):
                # This is an XML element, parse it directly
                return _parse_xml_element(msg._value_1)
            else:
                # Try standard serialization
                return serialize_object(msg)
        else:
            return serialize_object(m)
    except Exception:
        try:
            return serialize_object(m)
        except Exception:
            return None


def _collect_simple_items(obj):
    items = []
    def rec(x):
        if x is None:
            return
        if isinstance(x, dict):
            si = x.get('SimpleItem')
            if isinstance(si, list):
                for it in si:
                    if isinstance(it, dict):
                        name = it.get('Name', '')
                        value = it.get('Value')
                        items.append((str(name), value))
            elif isinstance(si, dict):
                name = si.get('Name', '')
                value = si.get('Value')
                items.append((str(name), value))
            # Also check for Source and Data containers
            for key in ['Source', 'Data']:
                if key in x:
                    rec(x[key])
            # Recurse into other dict values
            for k, v in x.items():
                if k not in ['SimpleItem', 'Source', 'Data']:
                    rec(v)
        elif isinstance(x, (list, tuple)):
            for it in x:
                rec(it)
    
    # Use our improved message serialization
    if hasattr(obj, 'Message') or hasattr(obj, 'Data') or hasattr(obj, 'Source'):
        # This looks like a message object
        serialized = _serialize_message(obj)
        rec(serialized)
    else:
        rec(obj)
    
    return items


def _try_parse_float_list(s):
    try:
        parts = [float(p) for p in re.split(r"[,;\s]+", str(s).strip()) if p != '']
        return parts if parts else None
    except Exception:
        return None


def extract_bboxes(m, name_tag='unknown'):
    """Extract bounding boxes from ONVIF analytics message.
    Returns a list of dicts: [{'l':..,'t':..,'r':..,'b':..,'normalized':True/False}]
    Heuristics: looks for keys like left/top/right/bottom or x/y/width/height,
    or comma-separated strings under keys containing 'box', 'rect', 'roi'.
    """
    _maybe_dump_raw_xml(m, name_tag, 'extract_bboxes')
    obj = _serialize_message(m)
    boxes = []

    def add_box(l, t, r, b, normalized=None):
        try:
            l = float(l); t = float(t); r = float(r); b = float(b)
        except Exception:
            return
        if normalized is None:
            # If numbers are <= 1.2, assume normalized [0..1]
            normalized = max(l, t, r, b) <= 1.2
        # Sanity: swap if needed
        if r < l:
            l, r = r, l
        if b < t:
            t, b = b, t
        boxes.append({'l': l, 't': t, 'r': r, 'b': b, 'normalized': normalized})

    def rec(x):
        if x is None:
            return
        if isinstance(x, dict):
            keys = {k.lower(): k for k in x.keys()}
            # If attributes captured under _attributes, merge a view for parsing
            attr_map = {}
            if '_attributes' in x and isinstance(x['_attributes'], dict):
                for ak, av in x['_attributes'].items():
                    attr_map[ak.lower()] = av
            # Pattern A: BoundingBox element style {'BoundingBox': {'_attributes': {'x':..,'y':..,'width':..,'height':..}}}
            if any(k in keys for k in ('boundingbox','bounding_box')):
                bb_key = keys.get('boundingbox') or keys.get('bounding_box')
                bb_val = x[bb_key]
                if isinstance(bb_val, dict):
                    attrs = bb_val.get('_attributes') if isinstance(bb_val.get('_attributes'), dict) else bb_val
                    if isinstance(attrs, dict):
                        xa = attrs.get('x'); ya = attrs.get('y'); wa = attrs.get('width'); ha = attrs.get('height')
                        if xa is not None and ya is not None and wa is not None and ha is not None:
                            try:
                                l = float(xa); t = float(ya)
                                r = l + float(wa); b = t + float(ha)
                                add_box(l, t, r, b)
                            except Exception:
                                pass
            # Pattern B: Shape -> Polygon -> Point list (compute bbox)
            if 'polygon' in keys:
                poly_key = keys['polygon']
                poly_val = x[poly_key]
                # Expect dict with possibly 'Point' list
                pts = []
                def collect_points(node):
                    if isinstance(node, dict):
                        if 'Point' in node:
                            pt = node['Point']
                            if isinstance(pt, list):
                                for p in pt:
                                    if isinstance(p, dict):
                                        attrs = p.get('_attributes') if isinstance(p.get('_attributes'), dict) else p
                                        try:
                                            px = float(attrs.get('x'))
                                            py = float(attrs.get('y'))
                                            pts.append((px, py))
                                        except Exception:
                                            pass
                            elif isinstance(pt, dict):
                                attrs = pt.get('_attributes') if isinstance(pt.get('_attributes'), dict) else pt
                                try:
                                    px = float(attrs.get('x'))
                                    py = float(attrs.get('y'))
                                    pts.append((px, py))
                                except Exception:
                                    pass
                        for v in node.values():
                            collect_points(v)
                    elif isinstance(node, list):
                        for it in node:
                            collect_points(it)
                collect_points(poly_val)
                if pts:
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    add_box(min(xs), min(ys), max(xs), max(ys))
            # Pattern 1: left/top/right/bottom
            if all(k in keys for k in ('left','top','right','bottom')):
                add_box(x[keys['left']], x[keys['top']], x[keys['right']], x[keys['bottom']])
            # Pattern 2: x/y/width/height
            elif all(k in keys for k in ('x','y','width','height')):
                l = x[keys['x']]; t = x[keys['y']]
                r = float(l) + float(x[keys['width']])
                b = float(t) + float(x[keys['height']])
                add_box(l, t, r, b)
            else:
                # Pattern 3: key contains box/rect/roi with comma numbers
                for lk, orig in keys.items():
                    if any(w in lk for w in ('box','rect','roi','rectangle','bbox','region')):
                        vals = _try_parse_float_list(x[orig])
                        if vals and len(vals) >= 4:
                            add_box(vals[0], vals[1], vals[2], vals[3])
                for v in x.values():
                    rec(v)
            # Recurse into attribute map in case it alone forms a box
            if attr_map and all(k in attr_map for k in ('x','y','width','height')):
                try:
                    l = float(attr_map['x']); t = float(attr_map['y'])
                    r = l + float(attr_map['width']); b = t + float(attr_map['height'])
                    add_box(l, t, r, b)
                except Exception:
                    pass
        elif isinstance(x, (list, tuple)):
            for it in x:
                rec(it)
        else:
            # Loose string with numbers
            vals = _try_parse_float_list(x)
            if vals and len(vals) >= 4:
                add_box(vals[0], vals[1], vals[2], vals[3])

    rec(obj)
    return boxes


def detect_object_label(m, topic_text, name_tag='unknown'):
    """Guess object label like Human, Vehicle, Face using SimpleItems or topic text."""
    _maybe_dump_raw_xml(m, name_tag, 'detect_object_label')
    obj = _serialize_message(m)
    items = _collect_simple_items(obj)
    
    # Debug output for object detection
    if DEBUG:
        print(f"    DEBUG detect_object_label: Found {len(items)} SimpleItems")
        for name, value in items:
            print(f"      SimpleItem: Name='{name}' Value='{value}'")
    
    for n, v in items:
        ln = (n or '').lower()
        lv = (str(v) if v is not None else '').lower()
        
        # Extended human detection keywords
        if any(w in ln or w in lv for w in ('human', 'person', 'people', 'pedestrian', 'man', 'woman', 'body')):
            if DEBUG:
                print(f"    DEBUG: Human detected via SimpleItem Name='{n}' Value='{v}'")
            return 'Human'
        
        # Vehicle detection
        if any(w in ln or w in lv for w in ('vehicle', 'car', 'truck', 'bus', 'bike', 'motorcycle', 'auto')):
            if DEBUG:
                print(f"    DEBUG: Vehicle detected via SimpleItem Name='{n}' Value='{v}'")
            return 'Vehicle'
        
        # Face detection
        if any(w in ln or w in lv for w in ('face', 'facial')):
            if DEBUG:
                print(f"    DEBUG: Face detected via SimpleItem Name='{n}' Value='{v}'")
            return 'Face'
        
        # Object type detection (some cameras use generic 'Object' field)
        if 'object' in ln and lv:
            if any(w in lv for w in ('human', 'person', 'people', 'pedestrian')):
                if DEBUG:
                    print(f"    DEBUG: Human detected via Object field Value='{v}'")
                return 'Human'
            if any(w in lv for w in ('vehicle', 'car', 'truck', 'bus')):
                if DEBUG:
                    print(f"    DEBUG: Vehicle detected via Object field Value='{v}'")
                return 'Vehicle'
        
        # Detection type fields
        if 'detection' in ln or 'detect' in ln:
            if any(w in lv for w in ('human', 'person', 'people')):
                if DEBUG:
                    print(f"    DEBUG: Human detected via Detection field Value='{v}'")
                return 'Human'
                
    # Topic text fallback with extended keywords
    tt = (topic_text or '').lower()
    if DEBUG:
        print(f"    DEBUG: Checking topic text: '{tt[:100]}...'")
    
    if re.search(r"\b(human|person|people|pedestrian|man|woman|body)\b", tt):
        if DEBUG:
            print(f"    DEBUG: Human detected via topic text")
        return 'Human'
    if re.search(r"\b(vehicle|car|truck|bus|bike|motorcycle|auto)\b", tt):
        if DEBUG:
            print(f"    DEBUG: Vehicle detected via topic text")
        return 'Vehicle'
    if re.search(r"\b(face|facial)\b", tt):
        if DEBUG:
            print(f"    DEBUG: Face detected via topic text")
        return 'Face'
    
    return None


def annotate_snapshot(img_bytes, boxes, label=None):
    """Draw boxes on image; return annotated JPEG bytes. Boxes in normalized or pixel units."""
    if not boxes or Image is None:
        return img_bytes
    try:
        im = Image.open(BytesIO(img_bytes)).convert('RGB')
        W, H = im.size
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for i, b in enumerate(boxes[:10]):  # cap to avoid spam
            l = float(b['l']); t = float(b['t']); r = float(b['r']); btm = float(b['b'])
            if b.get('normalized', False):
                l *= W; r *= W; t *= H; btm *= H
            # clamp
            l = max(0, min(W-1, l)); r = max(0, min(W-1, r))
            t = max(0, min(H-1, t)); btm = max(0, min(H-1, btm))
            color = (255, 0, 0)
            draw.rectangle([l, t, r, btm], outline=color, width=3)
            if font is not None:
                tag = f"{label or ''}".strip()
                if tag:
                    # small background box for readability
                    tw, th = draw.textsize(tag, font=font)
                    bg = [l, t - th - 2, l + tw + 4, t]
                    draw.rectangle(bg, fill=(255,0,0))
                    draw.text((l+2, t - th - 1), tag, font=font, fill=(255,255,255))
        out = BytesIO()
        im.save(out, format='JPEG', quality=85)
        return out.getvalue()
    except Exception:
        return img_bytes


def extract_source_data_kv(m):
    """Return (source_kv, data_kv) dicts from serialized ONVIF message."""
    obj = _serialize_message(m)
    src_kv, dat_kv = {}, {}
    try:
        msg = obj.get('Message', obj) if isinstance(obj, dict) else {}
        src = msg.get('Source', {}) if isinstance(msg, dict) else {}
        dat = msg.get('Data', {}) if isinstance(msg, dict) else {}
        for n, v in _collect_simple_items(src):
            if n:
                src_kv[str(n)] = '' if v is None else str(v)
        for n, v in _collect_simple_items(dat):
            if n:
                dat_kv[str(n)] = '' if v is None else str(v)
    except Exception:
        pass
    return src_kv, dat_kv


def extract_source_values(m):
    """Return list of raw Value fields from Message.Source.SimpleItem."""
    vals = []
    try:
        obj = _serialize_message(m)
        msg = obj.get('Message', obj) if isinstance(obj, dict) else {}
        src = msg.get('Source', {}) if isinstance(msg, dict) else {}
        for n, v in _collect_simple_items(src):
            if v is not None:
                vals.append(str(v))
    except Exception:
        pass
    return vals

class CamWorker(threading.Thread):
    def __init__(self, camcfg):
        super().__init__(daemon=True)
        self.cfg = camcfg
        self.last_fire = 0
        self.name_tag = camcfg.get("name") or camcfg["ip"]
        self.stop_flag = False
        self.event_props_dumped = False

    def _dump_event_properties(self, events):
        if self.event_props_dumped or not DUMP_EVENT_PROPERTIES:
            return
        try:
            props = events.GetEventProperties()
        except Exception as e:
            vlog(f"[{self.name_tag}] GetEventProperties failed: {e}")
            self.event_props_dumped = True
            return
        try:
            ser = serialize_object(props)
        except Exception:
            ser = None

        topic_paths = []
        def walk(node, path):
            if node is None:
                return
            if isinstance(node, dict):
                name = None
                if 'Name' in node and isinstance(node['Name'], str):
                    name = node['Name']
                elif 'name' in node and isinstance(node['name'], str):
                    name = node['name']
                new_path = path
                if name:
                    new_path = path + [name]
                    topic_paths.append('/'.join(new_path))
                for k, v in node.items():
                    if k.lower() in {'topic','topicset','subtopic'} or isinstance(v, (list, dict)):
                        walk(v, new_path)
            elif isinstance(node, list):
                for it in node:
                    walk(it, path)
        try:
            if isinstance(ser, dict) and 'TopicSet' in ser:
                walk(ser['TopicSet'], [])
        except Exception:
            pass
        try:
            Path(RAW_XML_DIR).mkdir(parents=True, exist_ok=True)
            base = Path(RAW_XML_DIR) / f"event_properties_{self.name_tag}"
            if ser is not None:
                with open(str(base)+'.json','w',encoding='utf-8') as fh:
                    json.dump(ser, fh, indent=2, ensure_ascii=False)
            if topic_paths:
                with open(str(base)+'_topics.txt','w',encoding='utf-8') as fh:
                    for p in sorted(set(topic_paths)):
                        fh.write(p+'\n')
            vlog(f"[{self.name_tag}] EventProperties dumped ({len(topic_paths)} topic paths)")
        except Exception as e:
            vlog(f"[{self.name_tag}] EventProperties dump error: {e}")
        self.event_props_dumped = True

    def run(self):
        ip = self.cfg["ip"]
        user = self.cfg["username"]
        pwd = self.cfg["password"]
        profile_index = int(self.cfg.get("profile", 0))

        while not self.stop_flag:
            try:
                cam = ONVIFCamera(ip, 80, user, pwd)
                media = cam.create_media_service()
                events = cam.create_events_service()
                devm = None
                try:
                    devm = cam.create_devicemgmt_service()
                except Exception:
                    pass

                profiles = get_profiles(media)
                if profile_index >= len(profiles):
                    profile_index = 0
                profile = profiles[profile_index]
                profile_token = profile.token

                # URIs
                snapshot_uri = get_snapshot_uri(media, profile_token)
                stream_uri = get_stream_uri(media, profile_token)

                # Replace stream with preferred substream if present
                if PREFER_STREAM and len(profiles) > PREFER_STREAM:
                    try:
                        sub_token = profiles[PREFER_STREAM].token
                        stream_uri = get_stream_uri(media, sub_token)
                    except Exception:
                        pass

                # Capability probe (optional, once per connect)
                if CAP_TEST:
                    try:
                        if devm is not None:
                            try:
                                info = devm.GetDeviceInformation()
                                vlog(f"[{self.name_tag}] Device: {getattr(info,'Manufacturer', '')} {getattr(info,'Model','')} FW {getattr(info,'FirmwareVersion','')} SN {getattr(info,'SerialNumber','')}")
                            except Exception as e:
                                vlog(f"[{self.name_tag}] GetDeviceInformation failed: {e}")
                            try:
                                dcaps = devm.GetCapabilities({'Category': 'All'})
                                ev_xaddr = getattr(getattr(dcaps, 'Events', object()), 'XAddr', None)
                                md_xaddr = getattr(getattr(dcaps, 'Media', object()), 'XAddr', None)
                                vlog(f"[{self.name_tag}] Capabilities XAddr: Events={ev_xaddr} Media={md_xaddr}")
                            except Exception as e:
                                vlog(f"[{self.name_tag}] GetCapabilities failed: {e}")
                        try:
                            caps = events.GetEventServiceCapabilities()
                            ws_pull = getattr(caps, 'WSPullPointSupport', None)
                            ws_base = getattr(caps, 'WSBaseNotificationSupport', None)
                            max_pp = getattr(caps, 'MaxPullPoints', None)
                            vlog(f"[{self.name_tag}] Event caps: WSPullPoint={ws_pull} WSBase={ws_base} MaxPullPoints={max_pp}")
                        except Exception as e:
                            vlog(f"[{self.name_tag}] GetEventServiceCapabilities failed: {e}")
                        # Dump event properties once
                        try:
                            self._dump_event_properties(events)
                        except Exception:
                            pass
                    except Exception as e:
                        vlog(f"[{self.name_tag}] Capability probe error: {e}")

                # Create PullPoint subscription (with capability check and fallbacks)
                try:
                    try:
                        caps = events.GetEventServiceCapabilities()
                        # Optional: log capabilities for troubleshooting
                        ws_pull = getattr(caps, 'WSPullPointSupport', None)
                        ws_base = getattr(caps, 'WSBaseNotificationSupport', None)
                        if ws_pull is False and ws_base is False:
                            raise RuntimeError("Camera reports no WS-Event pull or base-notification support")
                    except Exception:
                        pass

                    # Some cameras/firmware choke on create_type; passing dict is more compatible
                    vlog(f"[{self.name_tag}] Creating PullPoint subscription…")
                    sub_resp = events.CreatePullPointSubscription({
                        'InitialTerminationTime': 'PT1H'
                    })

                    # Derive PullPoint address if provided by SubscriptionReference
                    pull_addr = None
                    try:
                        pull_addr = sub_resp.SubscriptionReference.Address._value_1
                    except Exception:
                        pass
                    vlog(f"[{self.name_tag}] SubscriptionReference.Address={pull_addr}")

                    try:
                        pullpoint = cam.create_pullpoint_service(pull_addr) if pull_addr else cam.create_pullpoint_service()
                        vlog(f"[{self.name_tag}] PullPoint service created (addr={'provided' if pull_addr else 'default'})")
                    except Exception:
                        # Last resort: some devices accept PullMessages on the same events endpoint
                        vlog(f"[{self.name_tag}] PullPoint service creation failed, falling back to events service for PullMessages")
                        pullpoint = events

                    # Sync point to avoid missed/old events floods
                    try:
                        if hasattr(pullpoint, 'SetSynchronizationPoint'):
                            pullpoint.SetSynchronizationPoint()
                            vlog(f"[{self.name_tag}] SetSynchronizationPoint OK")
                    except Exception:
                        pass

                except Exception as e_sub:
                    raise RuntimeError(f"CreatePullPointSubscription failed: {e_sub}")

                print(f"[{self.name_tag}] ONVIF subscription OK. Listening for events…")
                while not self.stop_flag:
                    try:
                        # Try multiple request shapes to maximize compatibility
                        msgs = None
                        last_err = None

                        # 1) Dict payload
                        try:
                            vlog(f"[{self.name_tag}] PullMessages attempt A: dict payload via pullpoint")
                            msgs = pullpoint.PullMessages({'Timeout': 'PT30S', 'MessageLimit': 10})
                        except Exception as eA:
                            last_err = eA
                            vlog(f"[{self.name_tag}] PullMessages A(pullpoint) failed: {eA}")
                            try:
                                vlog(f"[{self.name_tag}] PullMessages attempt A2: dict payload via events")
                                msgs = events.PullMessages({'Timeout': 'PT30S', 'MessageLimit': 10})
                            except Exception as eA2:
                                last_err = eA2
                                vlog(f"[{self.name_tag}] PullMessages A2(events) failed: {eA2}")

                        # 2) Typed request from pullpoint/events
                        if msgs is None:
                            try:
                                vlog(f"[{self.name_tag}] PullMessages attempt B: typed request from pullpoint/events")
                                use_events_call = False
                                try:
                                    preq = pullpoint.create_type('PullMessages')
                                    use_events_call = False
                                except Exception:
                                    preq = events.create_type('PullMessages')
                                    use_events_call = True
                                try:
                                    preq.Timeout = 'PT30S'
                                except Exception:
                                    setattr(preq, 'Timeout', 'PT30S')
                                preq.MessageLimit = 10
                                if use_events_call:
                                    msgs = events.PullMessages(preq)
                                else:
                                    msgs = pullpoint.PullMessages(preq)
                            except Exception as eB:
                                last_err = eB
                                vlog(f"[{self.name_tag}] PullMessages B failed: {eB}")

                        # 3) kwargs (some wrappers accept kwargs)
                        if msgs is None:
                            try:
                                vlog(f"[{self.name_tag}] PullMessages attempt C: kwargs via pullpoint")
                                msgs = pullpoint.PullMessages(Timeout='PT30S', MessageLimit=10)
                            except Exception as eC:
                                last_err = eC
                                vlog(f"[{self.name_tag}] PullMessages C(pullpoint) failed: {eC}")
                                try:
                                    vlog(f"[{self.name_tag}] PullMessages attempt C2: kwargs via events")
                                    msgs = events.PullMessages(Timeout='PT30S', MessageLimit=10)
                                except Exception as eC2:
                                    last_err = eC2
                                    vlog(f"[{self.name_tag}] PullMessages C2(events) failed: {eC2}")

                        if msgs is None:
                            raise last_err or RuntimeError('PullMessages failed by all strategies')
                    except Fault as e:
                        if "wsa:MessageAddressingHeaderRequired" in str(e):
                            time.sleep(1)
                            continue
                        raise

                    if not msgs:
                        continue
                    # ONVIF typically returns msgs.NotificationMessage; keep backward compat with Msg
                    messages = getattr(msgs, 'NotificationMessage', None) or getattr(msgs, 'Msg', None)
                    if not messages:
                        continue
                    try:
                        vlog(f"[{self.name_tag}] Pulled {len(messages) if hasattr(messages,'__len__') else 'some'} messages")
                    except Exception:
                        pass

                    for m in messages:
                        # DEBUGGING: Always dump full event structure first
                        debug_dump_event(m, self.name_tag)
                        
                        # Build readable text from event for filtering
                        topic_txt = extract_event_text(m)
                        t_low = topic_txt.lower()
                        
                        # Ignore noisy events early (e.g., Bandwidth)
                        if IGNORE_FILTER and any(k in t_low for k in IGNORE_FILTER):
                            if DEBUG:
                                vlog(f"[{self.name_tag}] Ignored (ignore_filter match)")
                            continue
                        if DEBUG:
                            try:
                                short = topic_txt.replace('\n',' ')[:300]
                                vlog(f"[{self.name_tag}] Event topic: {short}")
                            except Exception:
                                pass
                        
                        # DEBUGGING: Temporarily disable topic filtering to see all events
                        # Comment out the filter check to see all events
                        # if TOPICS_FILTER and not any(k.lower() in t_low for k in TOPICS_FILTER):
                        #     if DEBUG:
                        #         try:
                        #             trig_dbg = None
                        #             try:
                        #                 trig_dbg = detect_trigger(m, topic_txt)
                        #             except Exception:
                        #                 pass
                        #             short = topic_txt.replace('\n',' ')[:300]
                        #             if trig_dbg:
                        #                 vlog(f"[{self.name_tag}] Skipped (no filter match) [{trig_dbg}]: {short}")
                        #             else:
                        #                 vlog(f"[{self.name_tag}] Skipped (no filter match): {short}")
                        #         except Exception:
                        #             pass
                        #     continue

                        # Always process event for debugging (ignore cooldown too)
                        # now = time.time()
                        # if now - self.last_fire < COOLDOWN_SECONDS:
                        #     continue
                        # self.last_fire = now

                        print(f"[{self.name_tag}] Processing event for analysis...")

                        # Detect trigger and object details
                        trig = 'Event'
                        obj_label = None
                        boxes = []
                        try:
                            trig = detect_trigger(m, topic_txt, self.name_tag)
                        except Exception:
                            pass
                        try:
                            obj_label = detect_object_label(m, topic_txt, self.name_tag)
                        except Exception:
                            pass
                        try:
                            boxes = extract_bboxes(m, self.name_tag)
                        except Exception:
                            pass

                        # Raw XML after full parsing (stage 'post_parse')
                        _maybe_dump_raw_xml(m, self.name_tag, 'post_parse')

                        # Debug output for analysis
                        print(f"[{self.name_tag}] ANALYSIS RESULTS:")
                        print(f"  Trigger detected: {trig}")
                        print(f"  Object label: {obj_label}")
                        print(f"  Bounding boxes: {len(boxes)} found")
                        if boxes:
                            for i, box in enumerate(boxes):
                                print(f"    Box {i+1}: {box}")
                        
                        # Extract and display all SimpleItem data
                        try:
                            src_kv, dat_kv = extract_source_data_kv(m)
                            print(f"  Source data: {src_kv}")
                            print(f"  Data fields: {dat_kv}")
                            src_vals = extract_source_values(m)
                            print(f"  Source values: {src_vals}")
                        except Exception as e:
                            print(f"  Error extracting data: {e}")

                        # Skip actual Telegram sending for debugging
                        print(f"[{self.name_tag}] Event analysis complete. (Telegram sending disabled for debugging)")
                        print(f"{'='*60}\n")

                time.sleep(1)

            except Exception as e:
                print(f"[{self.name_tag}] Error: {e}. Reconnecting in 10s...")
                time.sleep(10)

def main():
    cams = cfg["cameras"]
    workers = [CamWorker(c) for c in cams]
    for w in workers:
        w.start()
    print(f"Started {len(workers)} camera listeners.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()