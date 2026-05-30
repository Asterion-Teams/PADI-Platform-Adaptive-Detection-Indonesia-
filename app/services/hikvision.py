"""
Hikvision ISAPI Integration
----------------------------
Provides access to Hikvision camera features via ISAPI (HTTP Digest Auth):
  - Device info
  - High-resolution snapshot capture
  - PTZ control (if supported)
  - Event stream (motion, line crossing, etc.)

Usage:
    from app.services.hikvision import HikvisionISAPI

    cam = HikvisionISAPI("192.168.1.100", "admin", "password123")
    info = cam.get_device_info()
    jpg_bytes = cam.capture_snapshot(channel=1)
"""
from __future__ import annotations

import io
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from typing import Optional

import cv2
import numpy as np

try:
    import urllib.request
    import urllib.error
    from urllib.request import HTTPDigestAuthHandler, build_opener
except ImportError:
    pass


class HikvisionISAPI:
    """Client for Hikvision ISAPI (HTTP Digest Auth)."""

    def __init__(self, host: str, username: str = "admin", password: str = "",
                 port: int = 80, protocol: str = "http", timeout: int = 10):
        self.host = host.strip()
        self.username = username
        self.password = password
        self.port = int(port)
        self.protocol = protocol.strip().lower()
        self.timeout = int(timeout)
        self._base_url = f"{self.protocol}://{self.host}:{self.port}"
        self._opener = self._build_opener()

    def _build_opener(self):
        """Build urllib opener with Digest Auth."""
        auth_handler = HTTPDigestAuthHandler()
        auth_handler.add_password(
            realm="IP Camera",  # Hikvision default realm
            uri=self._base_url,
            user=self.username,
            passwd=self.password,
        )
        # Also add for other common realms
        auth_handler.add_password(
            realm="Hikvision",
            uri=self._base_url,
            user=self.username,
            passwd=self.password,
        )
        return build_opener(auth_handler)

    def _request(self, path: str, method: str = "GET", data: bytes = None,
                 content_type: str = None) -> bytes:
        """Make authenticated request to ISAPI endpoint."""
        url = f"{self._base_url}{path}"
        req = urllib.request.Request(url, data=data, method=method)
        if content_type:
            req.add_header("Content-Type", content_type)
        try:
            resp = self._opener.open(req, timeout=self.timeout)
            return resp.read()
        except urllib.error.HTTPError as e:
            raise HikvisionError(f"HTTP {e.code}: {e.reason} ({path})") from e
        except urllib.error.URLError as e:
            raise HikvisionError(f"Connection failed: {e.reason}") from e
        except Exception as e:
            raise HikvisionError(f"Request failed: {e}") from e

    def _parse_xml(self, xml_bytes: bytes) -> dict:
        """Parse Hikvision XML response into a flat dict."""
        result = {}
        try:
            root = ET.fromstring(xml_bytes)
            # Strip namespace prefix if present
            ns = re.match(r'\{.*\}', root.tag)
            ns_prefix = ns.group(0) if ns else ""
            for elem in root.iter():
                tag = elem.tag.replace(ns_prefix, "")
                if elem.text and elem.text.strip():
                    result[tag] = elem.text.strip()
        except ET.ParseError:
            pass
        return result

    # ------------------------------------------------------------------
    # Device Information
    # ------------------------------------------------------------------

    def get_device_info(self) -> dict:
        """Get camera device information.
        
        Returns dict with keys like:
            deviceName, deviceID, model, serialNumber,
            macAddress, firmwareVersion, encoderVersion, etc.
        """
        data = self._request("/ISAPI/System/deviceInfo")
        return self._parse_xml(data)

    def test_connection(self) -> dict:
        """Test if camera is reachable and credentials are valid.
        
        Returns:
            {"success": True, "device_name": "...", "model": "...", ...}
            or {"success": False, "error": "..."}
        """
        try:
            info = self.get_device_info()
            if info:
                return {
                    "success": True,
                    "device_name": info.get("deviceName", "Unknown"),
                    "model": info.get("model", "Unknown"),
                    "serial_number": info.get("serialNumber", ""),
                    "firmware": info.get("firmwareVersion", ""),
                    "mac_address": info.get("macAddress", ""),
                }
            return {"success": False, "error": "Empty response"}
        except HikvisionError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Snapshot Capture (High Resolution)
    # ------------------------------------------------------------------

    def capture_snapshot(self, channel: int = 1) -> Optional[bytes]:
        """Capture a high-resolution JPEG snapshot from the camera.
        
        This uses ISAPI to get a full-resolution frame (not the downscaled
        RTSP stream), which is much better for ANPR/OCR.
        
        Args:
            channel: Video channel number (1 = main, usually)
            
        Returns:
            JPEG bytes or None on failure.
        """
        # Try multiple known ISAPI snapshot paths
        paths = [
            f"/ISAPI/Streaming/channels/{channel}01/picture",
            f"/ISAPI/Streaming/channels/{channel}/picture",
            f"/Streaming/channels/{channel}01/picture",
            "/ISAPI/Streaming/channels/101/picture",
        ]
        for path in paths:
            try:
                data = self._request(path)
                # Verify it's actually JPEG
                if data and len(data) > 100 and data[:2] == b'\xff\xd8':
                    return data
            except HikvisionError:
                continue
        return None

    def capture_snapshot_cv2(self, channel: int = 1) -> Optional[np.ndarray]:
        """Capture snapshot and return as OpenCV BGR numpy array."""
        jpg = self.capture_snapshot(channel)
        if jpg is None:
            return None
        try:
            arr = np.frombuffer(jpg, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Time & Network
    # ------------------------------------------------------------------

    def get_time(self) -> dict:
        """Get camera system time."""
        try:
            data = self._request("/ISAPI/System/time")
            return self._parse_xml(data)
        except HikvisionError:
            return {}

    def get_network_interfaces(self) -> dict:
        """Get network interface info."""
        try:
            data = self._request("/ISAPI/System/Network/interfaces")
            return self._parse_xml(data)
        except HikvisionError:
            return {}

    # ------------------------------------------------------------------
    # Streaming Capabilities
    # ------------------------------------------------------------------

    def get_streaming_channels(self) -> list:
        """Get available streaming channels and their configurations."""
        try:
            data = self._request("/ISAPI/Streaming/channels")
            # Parse multiple channel entries
            channels = []
            root = ET.fromstring(data)
            ns = re.match(r'\{.*\}', root.tag)
            ns_prefix = ns.group(0) if ns else ""
            for ch_elem in root.iter(f"{ns_prefix}StreamingChannel"):
                ch = {}
                for elem in ch_elem:
                    tag = elem.tag.replace(ns_prefix, "")
                    if elem.text and elem.text.strip():
                        ch[tag] = elem.text.strip()
                if ch:
                    channels.append(ch)
            return channels
        except Exception:
            return []

    # ------------------------------------------------------------------
    # RTSP URL Builder
    # ------------------------------------------------------------------

    def get_rtsp_url(self, channel: int = 1, stream: int = 1) -> str:
        """Build RTSP URL for this camera.
        
        Args:
            channel: Camera channel (1-based)
            stream: 1=mainstream (HD), 2=substream (lower res)
        """
        stream_id = f"{channel}0{stream}"
        return f"rtsp://{self.username}:{self.password}@{self.host}:554/Streaming/Channels/{stream_id}"

    # ------------------------------------------------------------------
    # ANPR (License Plate Recognition) — DeepinView / iDS series only
    # ------------------------------------------------------------------

    def check_anpr_support(self) -> bool:
        """Check if this camera supports ANPR/LPR via ISAPI.
        
        Only Hikvision DeepinView (iDS-) and Traffic (iDS-TCM) cameras
        have built-in ANPR. Regular DS-2CD cameras will return 404.
        """
        try:
            data = self._request("/ISAPI/Traffic/channels/1/vehicleDetect/capabilities")
            return bool(data and len(data) > 50)
        except HikvisionError:
            return False

    def trigger_plate_recognition(self, channel: int = 1) -> Optional[dict]:
        """Trigger on-demand plate recognition (camera reads plate NOW).
        
        Only works on cameras with built-in ANPR (DeepinView/iDS series).
        
        Returns:
            {"plate": "B1234XYZ", "confidence": 0.95, "color": "blue", ...}
            or None if not supported or no plate visible.
        """
        try:
            data = self._request(
                f"/ISAPI/Traffic/channels/{channel}/vehicle/tryPlateRecognise",
                method="POST",
                data=b'<TryPlateRecognise><channel>1</channel></TryPlateRecognise>',
                content_type="application/xml",
            )
            if not data:
                return None
            parsed = self._parse_xml(data)
            plate_no = parsed.get("plateNo") or parsed.get("licensePlate") or ""
            if not plate_no:
                return None
            return {
                "plate": plate_no.strip().upper(),
                "confidence": float(parsed.get("confidence") or parsed.get("plateConfidence") or 0) / 100.0,
                "plate_color": parsed.get("plateColor", ""),
                "vehicle_color": parsed.get("vehicleColor", ""),
                "vehicle_type": parsed.get("vehicleType", ""),
                "direction": parsed.get("direction", ""),
            }
        except HikvisionError:
            return None

    def get_detected_plates(self, channel: int = 1, since: str = None,
                            max_results: int = 20) -> list:
        """Get list of plates detected by the camera's built-in ANPR.
        
        Args:
            channel: Video channel (usually 1)
            since: ISO timestamp to filter from (e.g. "2026-05-28T00:00:00Z")
            max_results: Maximum plates to return
            
        Returns:
            List of dicts: [{"plate": "B1234XY", "timestamp": "...", "confidence": 0.9, ...}]
        """
        if since is None:
            # Default: last 1 hour
            import datetime
            dt = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
            since = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        xml_body = f'<AfterTime><picTime>{since}</picTime></AfterTime>'
        
        try:
            data = self._request(
                f"/ISAPI/Traffic/channels/{channel}/vehicleDetect/plates",
                method="POST",
                data=xml_body.encode("utf-8"),
                content_type="application/xml",
            )
        except HikvisionError:
            return []

        if not data:
            return []

        plates = []
        try:
            root = ET.fromstring(data)
            ns = re.match(r'\{.*\}', root.tag)
            ns_prefix = ns.group(0) if ns else ""
            
            # Find all Plate or VehicleDetectResult elements
            for plate_elem in root.iter():
                tag = plate_elem.tag.replace(ns_prefix, "")
                if tag in ("Plate", "VehicleDetectResult", "PlateResult"):
                    entry = {}
                    for child in plate_elem:
                        child_tag = child.tag.replace(ns_prefix, "")
                        if child.text and child.text.strip():
                            entry[child_tag] = child.text.strip()
                    
                    plate_no = (entry.get("plateNo") or entry.get("licensePlate") 
                                or entry.get("plateNumber") or "")
                    if plate_no:
                        plates.append({
                            "plate": plate_no.strip().upper(),
                            "timestamp": entry.get("picTime") or entry.get("captureTime") or "",
                            "confidence": float(entry.get("confidence") or entry.get("plateConfidence") or 0) / 100.0,
                            "plate_color": entry.get("plateColor", ""),
                            "vehicle_color": entry.get("vehicleColor", ""),
                            "vehicle_type": entry.get("vehicleType", ""),
                            "lane": entry.get("laneNo", ""),
                            "speed": entry.get("carSpeed", ""),
                            "direction": entry.get("direction", ""),
                        })
                    if len(plates) >= max_results:
                        break
        except ET.ParseError:
            pass

        return plates

    def subscribe_anpr_events(self, callback=None, timeout: int = 30):
        """Subscribe to real-time ANPR event stream (alertStream).
        
        This opens a long-polling HTTP connection to receive plate detection
        events as they happen. Useful for real-time enforcement.
        
        Args:
            callback: Function called with each plate event dict
            timeout: Connection timeout in seconds
            
        Returns:
            List of events received within timeout (if no callback),
            or None (if callback is used, events are passed to callback).
            
        Note: This is blocking. Run in a separate thread for production use.
        """
        import urllib.request
        
        url = f"{self._base_url}/ISAPI/Event/notification/alertStream"
        req = urllib.request.Request(url)
        
        events = []
        try:
            resp = self._opener.open(req, timeout=timeout)
            # Read multipart stream
            buffer = b""
            start_time = time.time()
            while (time.time() - start_time) < timeout:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buffer += chunk
                
                # Parse ANPR events from multipart boundary
                while b"</EventNotificationAlert>" in buffer:
                    end_idx = buffer.find(b"</EventNotificationAlert>")
                    end_idx += len(b"</EventNotificationAlert>")
                    xml_chunk = buffer[:end_idx]
                    buffer = buffer[end_idx:]
                    
                    # Extract plate info from event
                    event = self._parse_anpr_event(xml_chunk)
                    if event:
                        if callback:
                            callback(event)
                        else:
                            events.append(event)
        except Exception:
            pass
        
        return events if not callback else None

    def _parse_anpr_event(self, xml_bytes: bytes) -> Optional[dict]:
        """Parse an ANPR event notification XML."""
        try:
            # Find the XML start
            start = xml_bytes.find(b"<EventNotificationAlert")
            if start == -1:
                return None
            xml_str = xml_bytes[start:].decode("utf-8", errors="ignore")
            root = ET.fromstring(xml_str)
            ns = re.match(r'\{.*\}', root.tag)
            ns_prefix = ns.group(0) if ns else ""
            
            parsed = {}
            for elem in root.iter():
                tag = elem.tag.replace(ns_prefix, "")
                if elem.text and elem.text.strip():
                    parsed[tag] = elem.text.strip()
            
            # Check if this is an ANPR event
            event_type = parsed.get("eventType", "")
            if "ANPR" not in event_type.upper() and "vehicle" not in event_type.lower():
                return None
            
            plate_no = (parsed.get("plateNo") or parsed.get("licensePlate") 
                        or parsed.get("plateNumber") or "")
            if not plate_no:
                return None
            
            return {
                "plate": plate_no.strip().upper(),
                "timestamp": parsed.get("dateTime") or parsed.get("picTime") or "",
                "confidence": float(parsed.get("confidence") or 0) / 100.0,
                "event_type": event_type,
                "channel": parsed.get("channelID") or parsed.get("dynChannelID") or "1",
                "plate_color": parsed.get("plateColor", ""),
                "vehicle_color": parsed.get("vehicleColor", ""),
                "vehicle_type": parsed.get("vehicleType", ""),
                "direction": parsed.get("direction", ""),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        return f"HikvisionISAPI({self.host}:{self.port}, user={self.username})"


class HikvisionError(Exception):
    """Error communicating with Hikvision camera."""
    pass


# ------------------------------------------------------------------
# Helper: ANPR with high-res snapshot
# ------------------------------------------------------------------

def capture_hires_for_anpr(host: str, username: str, password: str,
                           port: int = 80, channel: int = 1) -> Optional[np.ndarray]:
    """Convenience function: capture high-res snapshot for ANPR processing.
    
    Use this when a violation is detected to get a better image for plate reading
    than the low-res RTSP stream frame.
    """
    try:
        cam = HikvisionISAPI(host, username, password, port)
        return cam.capture_snapshot_cv2(channel)
    except Exception:
        return None
