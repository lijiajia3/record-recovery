"""Single INVENTED package resource interface; no model/task input option."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import traceback
import sccomics_supervised_runtime_v2 as runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True);p.add_argument('--expected-runtime-sha256',required=True)
    a=p.parse_args();out=Path(a.output_dir).absolute();assert not out.exists();out.mkdir(parents=True)
    source=Path(runtime.__file__);raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==a.expected_runtime_sha256
    (out/'runtime_before.py.txt').write_bytes(raw);(out/'helper_before.py.txt').write_bytes(Path(__file__).read_bytes())
    result={'identity':'INVENTED_IMPLEMENTER_RESOURCE_INTERFACE_CHECK_ONLY','actual_SC_annotation_weights_cache_prediction_read':False,
            'actual_runtime_or_model_backend_certified':False,'runtime_source_sha256':hashlib.sha256(raw).hexdigest()}
    with tempfile.TemporaryDirectory(prefix='INVENTED_runtime_resource_') as d:
        root=Path(d).resolve();package=root/'INVENTED_package';package.mkdir()
        init=package/'__init__.py';init.write_bytes(b"VALUE='INVENTED_package_only'\n")
        (package/'resource.txt').write_text('INVENTED RESOURCE ONLY')
        class InventedGuard:
            def capture(self,name):assert Path(name)==init;return init.read_bytes()
        try:
            loader=runtime.PythonLoader(InventedGuard(),str(init));reader=loader.get_resource_reader('INVENTED_package')
            assert reader.files()==package and reader.files().joinpath('resource.txt').read_text()=='INVENTED RESOURCE ONLY'
            assert reader.resource_path('resource.txt')==str(package/'resource.txt')
            result.update(passed=True,plain_fake_package_resource_interface=True)
        except Exception as e:result.update(passed=False,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
    assert source.read_bytes()==raw
    (out/'actual_result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True));return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
