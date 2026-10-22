"""
PCAP Reader & Packet Stream Parser
Fast, robust PCAP/PCAPNG ingestion supporting dpkt and scapy backends.
"""

import os
import socket
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def _inet_to_str(inet_bytes: bytes) -> str:
    """Convert raw 4-byte or 16-byte address to readable IP string."""
    try:
        if len(inet_bytes) == 4:
            return socket.inet_ntop(socket.AF_INET, inet_bytes)
        elif len(inet_bytes) == 16:
            return socket.inet_ntop(socket.AF_INET6, inet_bytes)
    except (ValueError, OSError):
        pass
    return "0.0.0.0"


class PCAPReader:
    """
    Streaming PCAP reader that parses raw packets and extracts IP/TCP/UDP metadata.
    """

    def __init__(self, pcap_path: str | Path, max_packets: Optional[int] = None):
        self.pcap_path = Path(pcap_path).resolve()
        self.max_packets = max_packets
        if not self.pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {self.pcap_path}")

    def iter_packets(self) -> Iterator[Dict[str, Any]]:
        """
        Yields individual parsed packet dictionaries using dpkt (primary) or scapy (fallback).
        """
        try:
            import dpkt
            yield from self._iter_packets_dpkt()
        except ImportError:
            logger.warning("dpkt not available; falling back to scapy reader.")
            yield from self._iter_packets_scapy()

    def _iter_packets_dpkt(self) -> Iterator[Dict[str, Any]]:
        import dpkt

        count = 0
        with open(self.pcap_path, "rb") as f:
            try:
                # Support standard pcap reader
                pcap = dpkt.pcap.Reader(f)
            except Exception:
                f.seek(0)
                try:
                    pcap = dpkt.pcapng.Reader(f)
                except Exception as e:
                    logger.error("Failed to read PCAP header with dpkt: %s", e)
                    return

            for ts, buf in pcap:
                if self.max_packets and count >= self.max_packets:
                    break

                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except Exception:
                    continue

                # Unpack IP layer
                ip = eth.data
                if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    continue

                proto = ip.p if hasattr(ip, "p") else 0
                src_ip = _inet_to_str(ip.src)
                dst_ip = _inet_to_str(ip.dst)

                sport, dport = 0, 0
                tcp_flags = 0
                tcp_win = 0
                hdr_len = 0

                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data
                    sport = tcp.sport
                    dport = tcp.dport
                    tcp_flags = tcp.flags
                    tcp_win = tcp.win
                    hdr_len = len(tcp) - len(tcp.data)
                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data
                    sport = udp.sport
                    dport = udp.dport
                    hdr_len = 8

                pkt_len = len(buf)
                count += 1

                yield {
                    "timestamp": float(ts),
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": int(sport),
                    "dst_port": int(dport),
                    "protocol": int(proto),
                    "packet_length": pkt_len,
                    "header_length": hdr_len,
                    "tcp_flags": tcp_flags,
                    "tcp_window": tcp_win,
                }

    def _iter_packets_scapy(self) -> Iterator[Dict[str, Any]]:
        from scapy.all import PcapReader, IP, TCP, UDP

        count = 0
        with PcapReader(str(self.pcap_path)) as reader:
            for pkt in reader:
                if self.max_packets and count >= self.max_packets:
                    break
                if not pkt.haslayer(IP):
                    continue

                ip = pkt[IP]
                proto = ip.proto
                src_ip = ip.src
                dst_ip = ip.dst
                sport, dport = 0, 0
                tcp_flags = 0
                tcp_win = 0
                hdr_len = 0

                if pkt.haslayer(TCP):
                    tcp = pkt[TCP]
                    sport = tcp.sport
                    dport = tcp.dport
                    tcp_flags = int(tcp.flags)
                    tcp_win = tcp.window
                    hdr_len = tcp.dataofs * 4 if tcp.dataofs else 20
                elif pkt.haslayer(UDP):
                    udp = pkt[UDP]
                    sport = udp.sport
                    dport = udp.dport
                    hdr_len = 8

                count += 1
                yield {
                    "timestamp": float(pkt.time),
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": int(sport),
                    "dst_port": int(dport),
                    "protocol": int(proto),
                    "packet_length": len(pkt),
                    "header_length": hdr_len,
                    "tcp_flags": tcp_flags,
                    "tcp_window": tcp_win,
                }

    def read_all(self) -> List[Dict[str, Any]]:
        return list(self.iter_packets())

