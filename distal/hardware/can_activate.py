"""Activate and configure CAN interfaces for Piper robot arms.

Each -p argument maps a USB port to a CAN interface name and bitrate.
Without -p, all detected interfaces are brought up at the default bitrate
keeping their existing names.

Example:
    python can_activate.py -p 1-1:1.0=can_arm1=1000000 -p 1-2:1.0=can_arm2=1000000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from pyroute2.iproute.linux import IPRoute

IFF_UP = 0x1


def get_can_interfaces(ipr: IPRoute) -> list[dict[str, Any]]:
    """Return all CAN-type network interfaces."""
    interfaces = []
    for link in ipr.get_links():
        linkinfo = link.get_attr("IFLA_LINKINFO")
        if not linkinfo or linkinfo.get_attr("IFLA_INFO_KIND") != "can":
            continue

        bitrate = 0
        info_data = linkinfo.get_attr("IFLA_INFO_DATA")
        if info_data:
            bittiming = info_data.get_attr("IFLA_CAN_BITTIMING")
            if bittiming:
                bitrate = bittiming.get("bitrate", 0)

        name = link.get_attr("IFLA_IFNAME")
        device_path = Path(f"/sys/class/net/{name}/device")
        bus_info = device_path.resolve().name if device_path.is_symlink() else None

        interfaces.append(
            {
                "index": link["index"],
                "name": name,
                "is_up": bool(link["flags"] & IFF_UP),
                "bitrate": bitrate,
                "bus_info": bus_info,
            }
        )
    return interfaces


def configure(
    ipr: IPRoute, iface: dict[str, Any], target_name: str, bitrate: int
) -> None:
    """Set bitrate, rename, and bring up a CAN interface."""
    idx = iface["index"]
    name = iface["name"]

    needs_reconfig = not iface["is_up"] or iface["bitrate"] != bitrate
    if needs_reconfig:
        ipr.link("set", index=idx, state="down")
        ipr.link("set", index=idx, kind="can", can_bitrate=bitrate)
        ipr.link("set", index=idx, state="up")
        idx = ipr.link_lookup(ifname=name)[0]

    if name != target_name:
        ipr.link("set", index=idx, state="down")
        ipr.link("set", index=idx, ifname=target_name)
        idx = ipr.link_lookup(ifname=target_name)[0]
        ipr.link("set", index=idx, state="up")

    print(f"  {name} ({iface['bus_info']}) -> {target_name} @ {bitrate}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Activate CAN interfaces.")
    parser.add_argument(
        "-p",
        "--port",
        dest="ports",
        action="append",
        metavar="USB=NAME=BITRATE",
        help="USB port mapping (repeatable), e.g. 1-1:1.0=can_arm1=1000000",
    )
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    args = parser.parse_args()

    subprocess.run(["modprobe", "gs_usb"], capture_output=True, check=False)

    with IPRoute() as ipr:
        interfaces = get_can_interfaces(ipr)
        if not interfaces:
            sys.exit("Error: no CAN interfaces detected")

        if args.ports:
            by_bus = {i["bus_info"]: i for i in interfaces}
            for entry in args.ports:
                parts = entry.split("=")
                if len(parts) != 3:
                    sys.exit(f"Error: invalid mapping '{entry}', use USB=NAME=BITRATE")
                usb_addr, name, bitrate = parts[0], parts[1], int(parts[2])
                if usb_addr not in by_bus:
                    print(f"  Warning: no interface at USB port '{usb_addr}', skipping")
                    continue
                configure(ipr, by_bus[usb_addr], name, bitrate)
        else:
            for iface in interfaces:
                configure(ipr, iface, iface["name"], args.bitrate)


if __name__ == "__main__":
