"""Offline API-contract tests; no real GitHub mutations or token required."""
import base64
import json
from pathlib import Path
import pytest
from tools.publish_github import MANIFEST, collect_files, blob_sha, publish

class Response:
    def __init__(self,status,data):self.status_code=status;self.data=data;self.content=b'x'
    def json(self):return self.data

class API:
    def __init__(self, files=None, empty=False, login='sharadphatak'):
        self.headers={};self.calls=[];self.files=files or {};self.empty=empty;self.login=login;self.blobs={}
    def request(self,method,url,json=None,timeout=None):
        route=url.replace('https://api.github.com','');self.calls.append((method,route,json))
        if route=='/user':return Response(200,{'login':self.login})
        if method=='GET' and route.endswith('/superkat_mlOps'):
            return Response(200,{'default_branch':'main','archived':False})
        if '/git/ref/' in route:
            return Response(409,{}) if self.empty else Response(200,{'object':{'sha':'parent'}})
        if method=='PUT' and '/contents/README.md' in route:
            self.empty=False;self.files['README.md']=base64.b64decode(json['content']);return Response(201,{})
        if method=='GET' and '/git/commits/' in route:return Response(200,{'tree':{'sha':'base'}})
        if method=='GET' and '/git/trees/' in route:
            return Response(200,{'tree':[{'path':p,'type':'blob','sha':blob_sha(b)} for p,b in self.files.items()]})
        if method=='GET' and '/git/blobs/' in route:
            sha=route.rsplit('/',1)[-1];data=next(b for b in self.files.values() if blob_sha(b)==sha)
            return Response(200,{'content':base64.b64encode(data).decode()})
        if method=='POST' and route.endswith('/git/blobs'):return Response(201,{'sha':'uploaded'})
        if method=='POST' and route.endswith('/git/trees'):return Response(201,{'sha':'newtree'})
        if method=='POST' and route.endswith('/git/commits'):return Response(201,{'sha':'newcommit'})
        if method=='PATCH' and '/git/refs/' in route:return Response(200,{})
        raise AssertionError((method,route))

@pytest.fixture
def project(tmp_path,monkeypatch):
    monkeypatch.setenv('GH_TOKEN','test-secret-not-real')
    for p in ['app.py','src/train.py','requirements.txt','.github/workflows/pipeline.yml','README.md']:
        file=tmp_path/p;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('project source')
    return tmp_path

def test_empty_existing_repository_initialized_and_updated(project):
    api=API(empty=True);result=publish(project,api)
    assert result['repository_url']=='https://github.com/sharadphatak/superkat_mlOps'
    assert any(m=='PUT' for m,r,p in api.calls)
    updates=[p for m,r,p in api.calls if m=='PATCH']
    assert updates==[{'sha':'newcommit','force':False}]

def test_repeat_upload_is_noop_and_receipt_excluded(project):
    files=collect_files(project);api=API(files=files)
    result=publish(project,api)
    assert result['files_updated']==0
    assert collect_files(project)==files
    assert not any(m=='POST' for m,r,p in api.calls)

def test_unrelated_files_preserved_and_conflicting_paths_rejected(project):
    api=API(files={'unrelated.txt':b'existing unrelated file'})
    publish(project,api)
    tree=next(p for m,r,p in api.calls if m=='POST' and r.endswith('/git/trees'))
    assert tree['base_tree']=='base'
    assert 'unrelated.txt' not in [x['path'] for x in tree['tree']]
    with pytest.raises(RuntimeError,match='Unmanaged remote file conflicts'):
        publish(project,API(files={'app.py':b'other application'}))

def test_wrong_account_rejected_before_remote_mutation(project):
    api=API(login='someone_else')
    with pytest.raises(RuntimeError,match='Use a token for sharadphatak'):publish(project,api)
    assert all(m=='GET' for m,r,p in api.calls)

def test_missing_repository_created_with_exact_name(project):
    class MissingAPI(API):
        def request(self,method,url,json=None,timeout=None):
            if method=='GET' and url.endswith('/repos/sharadphatak/superkat_mlOps'):
                self.calls.append((method,url,json));return Response(404,{})
            if method=='POST' and url.endswith('/user/repos'):
                self.calls.append((method,url,json));return Response(201,{'default_branch':'main'})
            return super().request(method,url,json,timeout)
    api=MissingAPI();publish(project,api)
    payload=next(p for m,r,p in api.calls if m=='POST' and r.endswith('/user/repos'))
    assert payload['name']=='superkat_mlOps' and payload['auto_init'] is True
