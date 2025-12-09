# Create the engine
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from ..db.traditional.models import Base, BioRule, CellType, CellTypeBioRule


engine = create_engine("sqlite:///bio_rules.db")

# Create tables if not exist
Base.metadata.create_all(engine)

# Session
Session = sessionmaker(bind=engine)
session = Session()

def fetch_rules(cell_types: list):
    query = session.execute(
        select(BioRule)
        .join(CellTypeBioRule, BioRule.id == CellTypeBioRule.bio_rule_id)
        .join(CellType, CellType.id == CellTypeBioRule.cell_type_id)
        .distinct(BioRule.id)
        .where(
            CellType.name.in_(cell_types)
        )
    )
    rules = query.scalars().all()
    
    return [{
        "rule": rule.rule,
        "rule_id": rule.id
    } for rule in rules]