"""Download and reconstruct original binary ZIPs with public URLs and hashes."""
from pathlib import Path
import argparse,json,hashlib,urllib.request,os

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 m=json.loads((Path(__file__).parent/'RELEASE_ASSET_MANIFEST.json').read_text());assert 'chunks' in m,'Use the finalized v1.0.0 source release'
 base='https://github.com/lijiajia3/scientific-record-recovery/releases/download/v1.0.0/'
 for name,desc in m['assets'].items():
  if not name.startswith('weights-caches-'):continue
  dst=a.out/name;partial=a.out/(name+'.downloading');assert not partial.exists(),'Preserve and inspect an earlier incomplete download'
  if dst.exists():
   h=hashlib.sha256()
   with dst.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
   assert dst.stat().st_size==desc['size_bytes'] and h.hexdigest()==desc['sha256'];continue
  whole=hashlib.sha256();size=0
  with partial.open('xb') as output:
   for row in sorted((x for x in m['chunks'] if x['zip']==name),key=lambda x:x['offset']):
    assert row['offset']==size;h=hashlib.sha256();n=0
    with urllib.request.urlopen(base+row['name'],timeout=120) as stream:
     for b in iter(lambda:stream.read(1048576),b''):output.write(b);h.update(b);whole.update(b);n+=len(b)
    assert n==row['size_bytes'] and h.hexdigest()==row['sha256'];size+=n
  assert size==desc['size_bytes'] and whole.hexdigest()==desc['sha256'];partial.rename(dst);print('VERIFIED',name,flush=True)
if __name__=='__main__':main()
