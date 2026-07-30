# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Persistent FNO/U-FNO inference server for unisolver Elder hybrid.

Load model **once**, then serve many Newton initial-guess requests over TCP.

Protocol (little-endian):
  Client -> Server:
    magic[4] = b"FNOQ"
    dt_sec     float64
    c[n]       float32  (n = NY*NX, row-major)
    P[n]       float32
  Server -> Client:
    success: magic b"FNOA" + c_pred[n] + P_pred[n]
    error:   magic b"FNOE" + uint32 msg_len + utf-8 message

Usage (start before TestElder with FnoPredictor.mode=persist):
  cd examples/cfd/elder_fno
  python fno_server.py --checkpoint outputs_elder_ufno/checkpoints --port 8765

Env on WSL (optional silence):
  export PYTHONWARNINGS=ignore::UserWarning:physicsnemo.distributed.manager
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import traceback

import numpy as np

from fno_infer_core import MAGIC_ERR, MAGIC_OK, MAGIC_REQ, FnoEngine


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed during recv")
        buf.extend(chunk)
    return bytes(buf)


def _handle_one(conn: socket.socket, engine: FnoEngine) -> None:
    magic = _recv_exact(conn, 4)
    if magic != MAGIC_REQ:
        raise ValueError(f"bad magic {magic!r}, expect {MAGIC_REQ!r}")
    (dt_sec,) = struct.unpack("<d", _recv_exact(conn, 8))
    n = engine.n_cells
    nbytes = 2 * n * 4
    raw = np.frombuffer(_recv_exact(conn, nbytes), dtype=np.float32).copy()
    c = raw[:n]
    P = raw[n:]
    try:
        c_pred, P_pred = engine.predict(c, P, float(dt_sec))
        out = MAGIC_OK + c_pred.tobytes() + P_pred.tobytes()
        conn.sendall(out)
        print(
            f"fno_server: ok dt={dt_sec:.6g}s "
            f"c[{c_pred.min():.3f},{c_pred.max():.3f}]",
            flush=True,
        )
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".encode("utf-8")
        conn.sendall(MAGIC_ERR + struct.pack("<I", len(msg)) + msg)
        print(f"fno_server: error {e}", flush=True)
        traceback.print_exc()


def main() -> None:
    ap = argparse.ArgumentParser(description="Persistent Elder FNO/U-FNO server")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--checkpoint",
        default="outputs_elder_ufno/checkpoints",
        help="checkpoint dir (latest .mdlus) or path used by load_checkpoint",
    )
    ap.add_argument(
        "--arch",
        default=None,
        help="fno | ufno (覆盖 config.yaml model.arch; 不传则用 config)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--device",
        default=None,
        help="cuda | cpu | default auto",
    )
    args = ap.parse_args()

    print(
        f"fno_server: loading model (once) config={args.config} "
        f"arch={args.arch} checkpoint={args.checkpoint} ...",
        flush=True,
    )
    engine = FnoEngine.load(
        args.config, args.checkpoint, device=args.device, arch=args.arch
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, int(args.port)))
    srv.listen(8)
    print(
        f"fno_server: listening on {args.host}:{args.port}  "
        f"n={engine.n_cells} ({engine.NY}x{engine.NX})  "
        f"arch={engine.arch}  device={engine.device}",
        flush=True,
    )
    print("fno_server: ready (Ctrl+C to stop)", flush=True)

    try:
        while True:
            conn, addr = srv.accept()
            with conn:
                try:
                    _handle_one(conn, engine)
                except Exception as e:
                    print(f"fno_server: connection {addr} failed: {e}", flush=True)
                    traceback.print_exc()
                    try:
                        msg = str(e).encode("utf-8")
                        conn.sendall(MAGIC_ERR + struct.pack("<I", len(msg)) + msg)
                    except Exception:
                        pass
    except KeyboardInterrupt:
        print("\nfno_server: shutdown", flush=True)
    finally:
        srv.close()


if __name__ == "__main__":
    main()
