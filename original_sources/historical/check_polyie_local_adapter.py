"""Meaningful invented-window checks: complete population, owner/negative mapping."""
import numpy as np
from polyie_local_adapter import candidate_batches, population_count, negative_sample, is_owner, record_for, key, chunk_encoding


def entity(i,role,start,end):return {'id':str(i),'type':role,'start':start,'end':end}
inp={'start':800,'entities':[entity(1,'CN',800,801),entity(2,'CN',999,1001),
    entity(3,'PN',810,811),entity(4,'PV',830,831),entity(5,'Condition',900,901),entity(6,'Condition',1100,1101)]}
rows=[r for b in candidate_batches(inp,2)for r in b]
assert len(rows)==population_count(inp)==4
assert set(rows)=={('1','3','4','6'),('2','3','4'),('2','3','4','5'),('2','3','4','6')}
positive={('2','3','4')};negative=negative_sample(inp,positive,50,np.random.default_rng(7))
assert set(negative)==set(rows)-positive
assert len({key(record_for(r))for r in rows})==4
first=dict(inp,start=0)
assert sum(len(b)for b in candidate_batches(first))==population_count(first)==6
assert is_owner(inp,[inp['entities'][1],inp['entities'][2],inp['entities'][3]])
class FakeTokenizer:
    cls_token_id=101
    sep_token_id=102
    def __call__(self,*args,**kwargs):
        return {'input_ids':[11,12,13],'offset_mapping':[(1,3),(4,8),(9,10)]}
fixture={'window_id':'invented','start':0,'tokens':['\ue103ll','test','x'],'text':'\ue103ll test x',
         'entities':[entity(10,'CN',0,1),entity(11,'PN',1,2),entity(12,'PV',2,3)]}
chunks,positions=chunk_encoding(fixture,FakeTokenizer())
assert positions=={'10':[(0,1,1.0)],'11':[(0,2,1.0)],'12':[(0,3,1.0)]}
assert chunks[0]['input_ids']==[101,11,12,13,102]
print('PASS full Cartesian, overlap ownership, negatives, strict IDs and stripped-leading-character mapping')
