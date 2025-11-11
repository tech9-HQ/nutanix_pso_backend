import pytest
from app.services.ranker import plan_suggestions
from app.models.schemas import SuggestPlanRequest

CASES = [
    ("VMware→NC2 Azure", "80 VMs migrated from VMware vSphere to Nutanix NC2 on Azure. DR setup required.",
     {"expect_phases":{"assessment","migration","dr"}, "expect_keywords":[("assessment","fitcheck"),("migration","nc2"),("dr","dr")] }),
    ("New NCI AHV site", "Deploy a new 6-node Nutanix NCI AHV cluster at Bangalore DC. DR not required.",
     {"expect_phases":{"deployment"}, "expect_keywords":[("deployment","deployment")] }),
    ("NDB modernization", "Migrate 25 Oracle databases from on-prem VMware to Nutanix Database Service (NDB) on AHV. Include FitCheck and DR replication.",
     {"expect_phases":{"assessment","database","dr"}, "expect_keywords":[("database","ndb"),("dr","dr")] }),
    ("NC2 AWS migration", "Move existing VMware workloads to Nutanix Cloud Clusters (NC2) on AWS. 100 VMs with DR.",
     {"expect_phases":{"assessment","migration","dr"}, "expect_keywords":[("migration","aws"),("dr","dr")] }),
    ("EUC on AHV", "Deploy Citrix VDI on Nutanix AHV for 500 users. Integrate with Nutanix Files and Flow.",
     {"expect_phases":{"assessment","deployment"}, "expect_keywords":[("deployment","euc")] }),
    ("Hybrid Azure + AHV", "Extend existing AHV cluster to Azure using NC2. Ensure DR protection between regions.",
     {"expect_phases":{"deployment","dr"}, "expect_keywords":[("deployment","nc2"),("dr","dr")] }),
    ("NKP", "Deploy Nutanix Kubernetes Platform (NKP) and integrate with NCI storage and Flow networking.",
     {"expect_phases":{"deployment"}, "expect_keywords":[("deployment","nkp")] }),
    ("NUS storage", "Migrate 300 TB of file data from NetApp to Nutanix Unified Storage (NUS). Require assessment and migration.",
     {"expect_phases":{"assessment","migration"}, "expect_keywords":[("migration","nus")] }),
    ("VMware DR Leap", "Enable Disaster Recovery for existing VMware workloads using Nutanix Leap and AHV target site.",
     {"expect_phases":{"dr"}, "expect_keywords":[("dr","leap")] }),
    ("Multisite enterprise", "Design and deploy a multisite Nutanix environment: 8 nodes per site across 3 sites. Include DR, database, and migration readiness.",
     {"expect_phases":{"assessment","deployment","migration","database","dr"}, "expect_keywords":[("deployment","multisite"),("database","ndb"),("dr","dr")] }),
]

@pytest.mark.parametrize("name, req_text, exp", CASES)
def test_golden(name, req_text, exp):
    req = SuggestPlanRequest(client_name="TestCo", requirements_text=req_text, top_k=5, limit=5)
    items, debug, journey = plan_suggestions(req)
    phases = { s.get("service_type","").lower() if s.get("service_type") else
               ("database" if (s.get("product_family")=="NDB") else None)
               for s in items }
    phases.discard(None)
    assert exp["expect_phases"].issubset(phases)

    # keyword sanity per phase
    for ph, kw in exp["expect_keywords"]:
        assert any(kw in (x.get("service_name","").lower()) for x in items), f"{name}: missing {kw} in {ph}"
