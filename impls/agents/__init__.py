from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.nm_crl import NonMarkovianCRLAgent
from agents.pcrl import ProbabilisticCRLAgent
from agents.nm_pcrl import NonMarkovianProbabilisticCRLAgent
from agents.dnce import DNCEAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    nm_crl=NonMarkovianCRLAgent,
    pcrl=ProbabilisticCRLAgent,
    nm_pcrl=NonMarkovianProbabilisticCRLAgent,
    dnce=DNCEAgent,
)
