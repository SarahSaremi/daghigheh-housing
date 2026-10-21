# Tiny localhost receiver: a relay page (GET /) gets files via postMessage from the browser tab
# and POSTs them back here; saved into Data/CBI_housing_raw/pdf
import http.server, urllib.parse, os
OUT = os.path.join("Data", "CBI_housing_raw", "pdf")
PAGE = b"""<html><body>relay<script>
window.addEventListener('message', async e => {
  const {name, buf} = e.data;
  const r = await fetch('/save?name=' + encodeURIComponent(name), {method:'POST', body: buf});
  e.source.postMessage({name, status: await r.text()}, '*');
});
if (window.opener) window.opener.postMessage({ready:true}, '*');
</script></body></html>"""
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(PAGE)
    def do_POST(self):
        name = os.path.basename(urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)["name"][0])
        data = self.rfile.read(int(self.headers["Content-Length"]))
        open(os.path.join(OUT, name), "wb").write(data)
        self.send_response(200); self.end_headers(); self.wfile.write(f"ok {len(data)}".encode())
http.server.HTTPServer(("127.0.0.1", 8765), H).serve_forever()
