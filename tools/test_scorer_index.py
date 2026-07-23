import os, sys
if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
    sys.exit("usage: %s <path-to-a-COPY-of-iris.db> -- never the live one: "
             "importing app runs schema migrations against IRIS_DB_PATH."
             % sys.argv[0])
os.environ["IRIS_DB_PATH"] = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import app as _app, api, iris_brain as brain
with _app.app.app_context(): df=brain.load_knowledge_base()

lab_key={lab:(src,str(cid).strip()) for lab,src,cid in
         zip(df.index, df["Source_Doc"], df["Clause_ID"])}
known=set(lab_key.values())

QUERIES=["the insurer shall ensure that the policyholder is informed of the claim settlement decision within the prescribed timeline",
 "charged to shareholders account","grievance redressal","free look period cancellation",
 "collect documents directly from hospitals","moratorium period non disclosure",
 "cashless claim settlement timeline","portability of health insurance policy",
 "pre existing disease waiting period","actl/ibnr free-look premium"]
fails=0
for q in QUERIES:
    kw=brain.get_clean_keywords(q); pl=re.sub(r"\s+"," ",q.lower()); cr=[]
    for raw,cl in kw:
        if " " in str(cl) or cl in brain.STOPWORDS_STRONG: continue
        r=brain.search_root(str(raw).lower(),cl)
        if r and r not in cr: cr.append(r)
    rk,rm={},{}
    for r in cr:
        labs=brain.clauses_with_root(r)
        rk[r]=None if labs is None else {lab_key[l] for l in labs if l in lab_key}
        sm=brain.sentence_masks(r)
        rm[r]=None if sm is None else {lab_key[l]:b for l,b in sm.items() if l in lab_key}
    # scorer WITH index vs scorer WITHOUT
    s_idx,_=api._make_scorer(pl,cr,True,run_len_cap=10,root_keys=rk,known_keys=known,root_masks=rm)
    s_ref,_=api._make_scorer(pl,cr,True,run_len_cap=10)
    bad=[]
    for lab,txt,src,cid in zip(df.index, df["Clause_Text"].fillna("").astype(str),
                               df["Source_Doc"], df["Clause_ID"]):
        m={"id":str(cid).strip(),"source":src,"raw_text":txt}
        a=s_idx(m); b=s_ref(dict(m))
        if a!=b: bad.append((lab,a,b))
    fails+=len(bad)
    idxed=sum(1 for r in cr if rm.get(r) is not None)
    print(f"  {'PASS' if not bad else 'FAIL'}  roots={len(cr):2d} indexed={idxed:2d}/{len(cr):2d}  {len(df)} clauses  {q[:44]}"
          + ("" if not bad else f"  e.g.{bad[:2]}"))
print(f"\n{'ALL PASS' if not fails else str(fails)+' MISMATCHES'}")
sys.exit(1 if fails else 0)
