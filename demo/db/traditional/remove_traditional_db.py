from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, BioRule, CellType, Patch, Trial

# Create the engine
engine = create_engine("sqlite:///bio_rules.db")

# Create tables if not exist
Base.metadata.create_all(engine)

# Session
Session = sessionmaker(bind=engine)
session = Session()

def delete_cell_rule(rule_id: int):
    rule = session.query(BioRule).get(rule_id)
    if rule:
        session.delete(rule)
        session.commit()
    else:
        print(f"BioRule with id {rule_id} not found.")
        
