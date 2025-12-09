from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from models import Base, BioRule, CellType, CellTypeBioRule, Patch, Trial

# Create the engine
engine = create_engine("sqlite:///bio_rules.db")

# Create tables if not exist
Base.metadata.create_all(engine)

# Session
Session = sessionmaker(bind=engine)
session = Session()

def insert_rule_traditional(rule: str, reference: str, weight: float, cell_type_id: int):

    cell_type = session.query(CellType).get(cell_type_id)
    if not cell_type:
        raise ValueError(f"No CellType with id {cell_type_id}")
    
    existing_rule = session.execute(
        select(BioRule)
        .filter(BioRule.rule == rule)
    ).scalars().first()
    
    if existing_rule and cell_type not in existing_rule.cell_types:
            existing_rule.cell_types.append(cell_type)
    else:
        new_rule = BioRule(
            rule=rule,
            reference=reference,
            weight=weight,
            cell_types=[cell_type]
        )
        session.add(new_rule)
    session.commit()
    return cell_type_id
    
def insert_cell_type(name: str):
    new_cell = CellType(name=name)
    session.add(new_cell)
    session.commit()
    return new_cell.id

def insert_trial(score: int, conditions: int, cell_type_ids: list, rule_ids: list):
    trial = Trial(score=score, conditions=conditions)

    # Link cell types
    for cid in cell_type_ids:
        cell_type = session.query(CellType).get(cid)
        if not cell_type:
            raise ValueError(f"CellType id {cid} not found")
        trial.cell_types.append(cell_type)

    # Link bio rules
    for rid in rule_ids:
        rule = session.query(BioRule).get(rid)
        if not rule:
            raise ValueError(f"BioRule id {rid} not found")
        trial.bio_rules.append(rule)

    session.add(trial)
    session.commit()
    return trial.id

def insert_patch(rule_id: int, trial_id: int, problem: str, fix: str):
    patch = Patch(
        rule_id=rule_id,
        trial_id=trial_id,
        problem_description=problem,
        attempted_fix=fix
    )
    session.add(patch)
    session.commit()
    return patch.id

insert_rule_traditional("T-cell spatial displacement (motility) should decrease over time for PD1+", "PMC4669840", 1, 1)
