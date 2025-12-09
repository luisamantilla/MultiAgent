from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class BioRule(Base):
    __tablename__ = 'bio_rules'
    
    id = Column(Integer, primary_key=True)
    rule = Column(String, nullable=False)
    reference = Column(String, nullable=False)
    weight = Column(Float, nullable=False)

    cell_types = relationship(
        'CellType',
        secondary='cell_types_bio_rules',
        back_populates='bio_rules'
    )
    
    trials = relationship(
        'Trial',
        secondary='bio_rules_trials',
        back_populates='bio_rules'
    )
    
    __table_args__ = (
        UniqueConstraint('rule', 'reference', name='uq_rule_reference'),
    )


class CellType(Base):
    __tablename__ = 'cell_types'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    bio_rules = relationship(
        'BioRule',
        secondary='cell_types_bio_rules',
        back_populates='cell_types'
    )

    trials = relationship(
        'Trial',
        secondary='cell_types_trials',
        back_populates='cell_types'
    )


class Trial(Base):
    __tablename__ = 'trials'
    
    id = Column(Integer, primary_key=True)
    score = Column(Integer, nullable=False)
    conditions = Column(Integer, nullable=False)

    cell_types = relationship(
        'CellType',
        secondary='cell_types_trials',
        back_populates='trials'
    )

    bio_rules = relationship(
        'BioRule',
        secondary='bio_rules_trials',
        back_populates='trials'
    )


class CellTypeBioRule(Base):
    __tablename__ = 'cell_types_bio_rules'
    
    bio_rule_id = Column(ForeignKey('bio_rules.id'), primary_key=True)
    cell_type_id = Column(ForeignKey('cell_types.id'), primary_key=True)


class CellTypeTrial(Base):
    __tablename__ = 'cell_types_trials'
    
    trial_id = Column(ForeignKey('trials.id'), primary_key=True)
    cell_type_id = Column(ForeignKey('cell_types.id'), primary_key=True)


class BioRuleTrial(Base):
    __tablename__ = 'bio_rules_trials'
    
    trial_id = Column(ForeignKey('trials.id'), primary_key=True)
    bio_rule_id = Column(ForeignKey('bio_rules.id'), primary_key=True)


class Patch(Base):
    __tablename__ = 'patches'
    
    id = Column(Integer, primary_key=True)
    rule_id = Column(ForeignKey('bio_rules.id'), nullable=False)
    trial_id = Column(ForeignKey('trials.id'), nullable=False)
    problem_description = Column(Text, nullable=True)
    attempted_fix = Column(Text, nullable=True)

    rule = relationship('BioRule')
    trial = relationship('Trial')
