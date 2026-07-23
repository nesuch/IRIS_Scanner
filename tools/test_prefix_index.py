import os, sys
if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
    sys.exit("usage: %s <path-to-a-COPY-of-iris.db> -- never the live one: "
             "importing app runs schema migrations against IRIS_DB_PATH."
             % sys.argv[0])
os.environ["IRIS_DB_PATH"] = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re, time, random
import app as _app, iris_brain as brain
with _app.app.app_context():
    df=brain.load_knowledge_base()

fails=0
def check(name, ok, detail=""):
    global fails
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  '+detail}")
    if not ok: fails+=1

lc=df["_ct_lc"].fillna("").astype(str); lcj=df["_ct_lcj"].fillna("").astype(str)
def regex_set(root):
    return {lab for lab in df.index if brain._root_present(root, lc[lab], lcj[lab])}

print(f"index: vocab={len(brain._IDX_WORDS)} postings={sum(len(v) for v in brain._IDX_MAP.values())}")

print("\n[1] equivalence vs regex, real query roots")
ROOTS=["insur","shall","ensur","policyhold","inform","claim","settlement","decis","within",
       "prescrib","timelin","premium","grievanc","redress","cashless","moratorium","hospital",
       "de","a","zzzznotpresent","tpa","ibnr","health","account","fund","shareholder"]
bad=[r for r in ROOTS if brain.clauses_with_root(r)!=regex_set(r)]
check(f"{len(ROOTS)} curated roots identical", not bad, str(bad))

print("\n[2] equivalence on 400 random vocabulary words + prefixes")
random.seed(7)
sample=random.sample(list(brain._IDX_WORDS), 300)
sample+=[w[:k] for w in random.sample(list(brain._IDX_WORDS),100) for k in (2,3,5) if len(w)>k]
bad=[w for w in sample if brain.clauses_with_root(w)!=regex_set(w)]
check(f"{len(sample)} random roots/prefixes identical", not bad, str(bad[:5]))

print("\n[3] guard: non-\\w roots must return None (regex fallback)")
check("actl/ibnr -> None", brain.clauses_with_root("actl/ibnr") is None)
check("free-look -> None", brain.clauses_with_root("free-look") is None)
check("empty -> None", brain.clauses_with_root("") is None)
check("uppercase root normalised", brain.clauses_with_root("INSUR")==regex_set("insur"))

print("\n[4] invalidation via set_clause_text")
lab=df.index[5]; orig=str(df.loc[lab,"Clause_Text"])
brain.set_clause_text(df.index==lab, "zqxwv unicornium de-empanelment protocol")
ok1 = lab in (brain.clauses_with_root("zqxwv") or set())
ok2 = brain.clauses_with_root("unicornium")=={lab}
ok3 = lab in (brain.clauses_with_root("deempanel") or set())   # hyphen-collapsed form indexed
check("new words indexed", ok1 and ok2, f"{ok1} {ok2}")
check("hyphen-collapsed form indexed", ok3)
gone = brain.clauses_with_root("zqxwv")
brain.set_clause_text(df.index==lab, orig)
check("old words dropped on revert", not brain.clauses_with_root("zqxwv"))
check("original restored + reindexed", brain.clauses_with_root("insur")==regex_set("insur"))
check("full-corpus consistency after edit",
      all(brain.clauses_with_root(r)==regex_set(r) for r in ROOTS[:12]))

print("\n[5] speed")
t0=time.perf_counter()
for _ in range(50):
    for r in ROOTS[:11]: brain.clauses_with_root(r)
t_i=(time.perf_counter()-t0)/50
t0=time.perf_counter()
for r in ROOTS[:11]: regex_set(r)
t_r=time.perf_counter()-t0
print(f"  11 roots over {len(df)} clauses:  regex {t_r*1000:7.1f} ms   index {t_i*1000:6.2f} ms  ({t_r/t_i:.0f}x)")

print(f"\n{'ALL PASS' if not fails else str(fails)+' FAILURES'}")
sys.exit(1 if fails else 0)
