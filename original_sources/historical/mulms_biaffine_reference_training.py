"""Reuse the audited main fitting function with an explicitly frozen reference.

Only architecture/class/output/freeze pointers are replaced. Scientific data,
negative draws, development populations, thresholds, optimizer and budget stay
in the unchanged original function. Never overwrites the primary relation run.
"""
import mulms_local_relation_training as trainer
from mulms_biaffine_reference import BiaffineReference

REFERENCE_DEST=trainer.OUTPUT/'biaffine_reference'
REFERENCE_FREEZE=trainer.ROOT/'research/mulms_biaffine_reference_training_freeze.json'


def main():
    trainer.RelationHead=BiaffineReference
    trainer.ARCHITECTURES=('biaffine',)
    trainer.DEST=REFERENCE_DEST
    trainer.FREEZE=REFERENCE_FREEZE
    trainer.main()
    import json
    state_path=trainer.DEST/'training_state.json';state=json.loads(state_path.read_text())
    assert state['status']=='completed_all_three_architectures'and state['architectures']==['biaffine']
    state['status']='completed_biaffine_reference_three_seeds';trainer.write(state_path,state)
    path=trainer.DEST/'complete_selections.json';complete=json.loads(path.read_text())
    assert set(complete['selections'])=={'biaffine'};complete['status']='reference_architecture_complete';trainer.write(path,complete)


if __name__=='__main__':main()
