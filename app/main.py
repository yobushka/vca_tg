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
    # Message payload (SimpleItem Name/Value are here)
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

    return " ".join([p for p in parts if isinstance(p, str) and p])


def _to_boolish(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "on"}


def detect_trigger(m, topic_text):
    """Best-effort detection of the fired trigger (e.g., Motion, Tripwire).
    Looks into Message.Data/Source SimpleItem Name/Value pairs and topic text.
    Returns a short label like 'Motion', 'Intrusion', 'CrossLine', 'Human', etc.
    """
    try:
        obj = serialize_object(getattr(m, 'Message', m))
    except Exception:
        obj = None

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

    # Heuristics based on SimpleItem Name/Value
    name_val = [(n or "", v) for (n, v) in simple_items]
    for n, v in name_val:
        ln = n.lower()
        if _to_boolish(v) or (v is None and any(k in ln for k in ('motion','trip','cross','intrusion','tamper','audio','human','person','vehicle'))):
            # Motion
            if 'motion' in ln or ln in {'ismotion','motion'}:
                return 'Motion'
            # Intrusion
            if 'intrusion' in ln:
                return 'Intrusion'
            # Trip/Cross/Line
            if 'trip' in ln and 'line' in ln:
                return 'Tripwire'
            if 'cross' in ln and 'line' in ln:
                return 'CrossLine'
            if 'line' in ln:
                return 'Line'
            if 'cross' in ln:
                return 'Cross'
            if 'trip' in ln:
                return 'Trip'
            # Human/Person
            if 'human' in ln or 'person' in ln or 'people' in ln:
                return 'Human'
            # Vehicle
            if 'vehicle' in ln or 'car' in ln or 'truck' in ln or 'bus' in ln or 'bike' in ln:
                return 'Vehicle'
            # Tamper
            if 'tamper' in ln:
                return 'Tamper'
            # Audio
            if 'audio' in ln or 'sound' in ln:
                return 'Audio'

    # Topic path fallback
    tt = (topic_text or '').lower()
    m = re.search(r"\b(motion|intrusion|tripwire|crossline|cross|line|human|person|vehicle|tamper|audio)\b", tt)
    if m:
        val = m.group(1)
        # Normalize
        mapping = {
            'cross': 'Cross',
            'line': 'Line',
            'person': 'Human',
        }
        return mapping.get(val, val.capitalize())

    # Rule name could hint (e.g., MyMotionDetectorRule)
    rm = re.search(r"\b([A-Za-z]*Motion[A-Za-z]*)\b", tt)
    if rm:
        return 'Motion'

    return 'Event'


def _serialize_message(m):
    try:
        return serialize_object(getattr(m, 'Message', m))
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
                        items.append((str(it.get('Name', '')), it.get('Value')))
            elif isinstance(si, dict):
                items.append((str(si.get('Name', '')), si.get('Value')))
            for v in x.values():
                rec(v)
        elif isinstance(x, (list, tuple)):
            for it in x:
                rec(it)
    rec(obj)
    return items


def _try_parse_float_list(s):
    try:
        parts = [float(p) for p in re.split(r"[,;\s]+", str(s).strip()) if p != '']
        return parts if parts else None
    except Exception:
        return None


def extract_bboxes(m):
    """Extract bounding boxes from ONVIF analytics message.
    Returns a list of dicts: [{'l':..,'t':..,'r':..,'b':..,'normalized':True/False}]
    Heuristics: looks for keys like left/top/right/bottom or x/y/width/height,
    or comma-separated strings under keys containing 'box', 'rect', 'roi'.
    """
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


def detect_object_label(m, topic_text):
    """Guess object label like Human, Vehicle, Face using SimpleItems or topic text."""
    obj = _serialize_message(m)
    items = _collect_simple_items(obj)
    for n, v in items:
        ln = (n or '').lower()
        lv = (str(v) if v is not None else '').lower()
        if any(w in ln or w in lv for w in ('human', 'person', 'people')):
            return 'Human'
        if any(w in ln or w in lv for w in ('vehicle', 'car', 'truck', 'bus', 'bike')):
            return 'Vehicle'
        if any(w in ln or w in lv for w in ('face',)):
            return 'Face'
    tt = (topic_text or '').lower()
    if re.search(r"\b(human|person|people)\b", tt):
        return 'Human'
    if re.search(r"\b(vehicle|car|truck|bus|bike)\b", tt):
        return 'Vehicle'
    if re.search(r"\bface\b", tt):
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
                        if TOPICS_FILTER and not any(k.lower() in t_low for k in TOPICS_FILTER):
                            if DEBUG:
                                try:
                                    trig_dbg = None
                                    try:
                                        trig_dbg = detect_trigger(m, topic_txt)
                                    except Exception:
                                        pass
                                    short = topic_txt.replace('\n',' ')[:300]
                                    if trig_dbg:
                                        vlog(f"[{self.name_tag}] Skipped (no filter match) [{trig_dbg}]: {short}")
                                    else:
                                        vlog(f"[{self.name_tag}] Skipped (no filter match): {short}")
                                except Exception:
                                    pass
                            continue

                        now = time.time()
                        if now - self.last_fire < COOLDOWN_SECONDS:
                            continue
                        self.last_fire = now

                        # Detect trigger and object details
                        trig = 'Event'
                        obj_label = None
                        boxes = []
                        try:
                            trig = detect_trigger(m, topic_txt)
                        except Exception:
                            pass
                        try:
                            obj_label = detect_object_label(m, topic_txt)
                        except Exception:
                            pass
                        try:
                            boxes = extract_bboxes(m)
                        except Exception:
                            pass

                        ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
                        details = []
                        if obj_label:
                            details.append(obj_label)
                        if boxes:
                            details.append(f"boxes={len(boxes)}")
                        try:
                            src_kv, dat_kv = extract_source_data_kv(m)
                            if 'Rule' in dat_kv:
                                details.append(f"rule={dat_kv['Rule']}")
                            if 'VideoAnalyticsConfigurationToken' in src_kv:
                                details.append(f"analytics={src_kv['VideoAnalyticsConfigurationToken']}")
                            if 'VideoSourceConfigurationToken' in src_kv:
                                details.append(f"source={src_kv['VideoSourceConfigurationToken']}")
                            src_vals = extract_source_values(m)
                            if src_vals:
                                details.append(f"srcval={src_vals[0]}")
                        except Exception:
                            pass
                        extra = (" ".join(details)) if details else ""
                        caption = f"📹 {self.name_tag} [{trig}] @ {ts} {extra}\n{topic_txt[:180]}".strip()
                        print(f"[{self.name_tag}] Event matched → sending Telegram...")

                        # Snapshot
                        snap_url = sanitize_snapshot_uri(snapshot_uri, ip)
                        try:
                            # Try Basic first
                            r = requests.get(snap_url, timeout=10, stream=True, auth=HTTPBasicAuth(user, pwd))
                            if r.status_code == 401:
                                # Fallback to Digest
                                r = requests.get(snap_url, timeout=10, stream=True, auth=HTTPDigestAuth(user, pwd))
                            r.raise_for_status()
                            img = r.content
                            # annotate if we have boxes
                            try:
                                img = annotate_snapshot(img, boxes, label=obj_label or trig)
                            except Exception:
                                pass
                            tg_send_photo(caption, img, "snapshot.jpg")
                        except Exception as e:
                            print(f"[{self.name_tag}] snapshot failed: {e}")

                        # Video clip
                        try:
                            # Inject credentials in RTSP if missing
                            if "@" not in stream_uri and user:
                                parsed = urlparse(stream_uri)
                                netloc = parsed.netloc
                                if not netloc:
                                    # some cams return only path; build from IP
                                    netloc = f"{ip}:554"
                                if ":" not in netloc:
                                    netloc += ":554"
                                netloc = f"{user}:{pwd}@{netloc}"
                                stream_uri = urlunparse((parsed.scheme or "rtsp", netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
                            out = f"/app/clip_{self.name_tag}_{int(now)}.mp4"
                            ffmpeg_record(stream_uri, CLIP_SECONDS, out)
                            tg_send_video(caption, out)
                            try:
                                os.remove(out)
                            except OSError:
                                pass
                        except Exception as e:
                            print(f"[{self.name_tag}] video failed: {e}")

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