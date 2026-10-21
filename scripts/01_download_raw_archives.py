"""Download ORIGINAL raw HBS archives (data.rar, MS-Access inside) from SCI mirror, untouched.
Stored in Data/HBSIR/1_original/<year>/. No processing."""
import logging, sys
from bssir import archive_handler
import hbsir
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
api = hbsir.api
years = [int(a) for a in sys.argv[1:]] or list(range(1363, 1403))
for y in years:
    for src in ("mirror", "original"):
        try:
            archive_handler.download([y], lib_metadata=api.metadata, lib_defaults=api.defaults,
                                     replace=False, source=src)
            print("OK", y, src, flush=True); break
        except Exception as e:
            print("FAIL", y, src, repr(e)[:200], flush=True)
