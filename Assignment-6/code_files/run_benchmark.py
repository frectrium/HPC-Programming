#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_benchmark.py - Compile, verify, and benchmark PIC interpolation (Python 2.7.5)

Expects in CWD: main.cpp utils.cpp utils.h init.cpp init.h input_file_maker.cpp
                Test_input.bin  Test_Mesh.out

Outputs: outputs/*.csv

Usage:  python run_benchmark.py          # full
        python run_benchmark.py --quick  # fast sanity-check
"""
from __future__ import print_function
import subprocess, os, sys, re, csv, time, shutil, multiprocessing

# -- Config -----------------------------------------------------------
CONFIGS = [
    ('a',250,100,900000,10),('b',250,100,5000000,10),
    ('c',500,200,3600000,10),('d',500,200,20000000,10),
    ('e',1000,400,14000000,10),
]
METHODS = [
    ('serial',0,False),('atomic',1,True),
    ('reduction',2,True),('reduction_unrolled',3,True),
]
PARALLEL_METHODS = [m for m in METHODS if m[2]]
THREAD_COUNTS = [1,2,4,8,16]
NUM_RUNS = 3
CXX = 'g++'; CXXFLAGS = '-O3 -mavx2 -mfma -std=c++11'
SRC = 'main.cpp utils.cpp init.cpp'
ODIR = 'outputs'; TEST_IN = 'Test_input.bin'; TEST_OUT = 'Test_Mesh.out'
NCPU = multiprocessing.cpu_count()

# -- Helpers ----------------------------------------------------------
def run_cmd(cmd, stdin_data=None, env=None):
    p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    o, e = p.communicate(input=stdin_data)
    return p.returncode, o, e

def parse_time(out):
    m = re.search(r'interpolation time.*?=\s*([\d.]+)\s*seconds', out)
    return float(m.group(1)) if m else None

def make_env(thr, pin=True):
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(thr)
    if pin and thr <= NCPU:
        env['GOMP_CPU_AFFINITY'] = '0-%d' % (thr-1)
    return env

def mesh_match(fa, fb, atol=1e-6):
    try:
        la = open(fa).readlines(); lb = open(fb).readlines()
    except IOError:
        return False, False, float('inf'), -1
    if len(la) != len(lb): return False, False, float('inf'), -1
    exact, mx, nd = True, 0.0, 0
    for a, b in zip(la, lb):
        va, vb = a.split(), b.split()
        if len(va) != len(vb): return False, False, float('inf'), -1
        for sa, sb in zip(va, vb):
            if sa != sb: exact = False
            d = abs(float(sa)-float(sb))
            if d > 0: nd += 1
            if d > mx: mx = d
    return exact, mx <= atol, mx, nd

def ensured(path):
    try: os.makedirs(path)
    except OSError:
        if not os.path.isdir(path): raise

def fmtpts(n):
    if n >= 1000000: return '%.1fM' % (n/1000000.0)
    if n >= 1000: return '%.0fK' % (n/1000.0)
    return str(n)

# -- Check files ------------------------------------------------------
def check_files():
    need = ['main.cpp','utils.cpp','utils.h','init.cpp','init.h',
            'input_file_maker.cpp', TEST_IN, TEST_OUT]
    miss = [f for f in need if not os.path.exists(f)]
    if miss:
        print('ERROR -- missing: ' + ', '.join(miss)); sys.exit(1)
    print('All source + test files found.  (%d CPU cores detected)' % NCPU)

# -- Compile ----------------------------------------------------------
def compile_all():
    print('\n' + '='*65 + '\nCOMPILING\n' + '='*65)
    rc,_,e = run_cmd('%s %s -o input_file_maker input_file_maker.cpp' % (CXX, CXXFLAGS))
    if rc: print('FAIL:\n%s' % e); sys.exit(1)
    print('  input_file_maker OK')
    bins = {}
    for label, mid, omp in METHODS:
        b = 'interp_%s' % label
        fl = '%s -DMETHOD=%d' % (CXXFLAGS, mid)
        if omp: fl += ' -fopenmp'
        cmd = '%s %s -o %s %s' % (CXX, fl, b, SRC)
        print('  %-24s %s' % (label, cmd))
        rc,_,e = run_cmd(cmd)
        if rc: print('  FAIL:\n%s' % e); sys.exit(1)
        bins[label] = b
    print('  All OK\n')
    return bins

# -- Verify -----------------------------------------------------------
def verify_test(bins, tcounts):
    print('='*65 + '\nVERIFICATION (against %s)\n' % TEST_OUT + '='*65)
    vtc = sorted(set([1]+[t for t in tcounts if t<=NCPU]))
    ver = []; wrong = False
    for label, _, omp in METHODS:
        tc = [1] if not omp else vtc
        for t in tc:
            env = make_env(t, pin=False)
            rc,out,_ = run_cmd('./%s %s' % (bins[label], TEST_IN), env=env)
            if rc:
                st='RUN_FAIL'; ex=False; md=-1.0; nc=-1
            else:
                ex,tok,md,nc = mesh_match('Mesh.out', TEST_OUT)
                st = 'EXACT_MATCH' if ex else ('TOL_MATCH' if tok else 'FAIL')
            tag = '%-22s thr=%2d' % (label, t)
            if st=='EXACT_MATCH':   print('  %s  PASS (exact)' % tag)
            elif st=='TOL_MATCH':   print('  %s  PASS (tol, d=%.2e)' % (tag, md))
            elif st=='RUN_FAIL':    print('  %s  SKIP (run failed)' % tag)
            else:                   print('  %s  **FAIL** d=%.2e' % (tag, md)); wrong=True
            ver.append(dict(method=label,threads=t,status=st,exact=ex,max_diff=md,cells_differ=nc))
    if wrong:
        print('\n  *** WRONG OUTPUT -- aborting. ***'); save_ver(ver); sys.exit(1)
    print('  All OK\n')
    return ver

def save_ver(ver):
    ensured(ODIR)
    p = os.path.join(ODIR,'verification_results.csv')
    with open(p,'w') as f:
        w=csv.writer(f)
        w.writerow(['method','threads','status','exact_match','max_diff','cells_differ'])
        for v in ver:
            w.writerow([v['method'],v['threads'],v['status'],v['exact'],
                        '%.2e'%v['max_diff'],v['cells_differ']])
    print('  Written: %s' % p)

# -- Generate inputs --------------------------------------------------
def gen_input(cfg):
    nm,nx,ny,np_,mi = cfg
    of = 'input_%s.bin' % nm
    if os.path.exists(of): print('\n  [%s] exists -- skip' % nm); return
    print('\n  [%s] NX=%d NY=%d pts=%s ...' % (nm,nx,ny,fmtpts(np_)), end=' ')
    sys.stdout.flush()
    rc,_,e = run_cmd('./input_file_maker', stdin_data='%d %d\n%d\n%d\n' % (nx,ny,np_,mi))
    if rc: print('FAIL\n%s' % e); sys.exit(1)
    os.rename('input.bin', of)
    print('OK (%.1f MB)' % (os.path.getsize(of)/1048576.0))

# -- Benchmark --------------------------------------------------------
def do_benchmark(bins, cfgs, tcounts, nruns):
    atc = [t for t in tcounts if t<=NCPU]
    if atc != tcounts:
        print('  NOTE: only %d cores, using threads %s' % (NCPU, atc))
    print('='*65)
    print('BENCHMARKING (%d cfgs x %d methods x <=%d thr x %d runs)' %
          (len(cfgs), len(METHODS), len(atc), nruns))
    print('='*65)
    res=[]; xchk=[]
    total=sum(len([1] if not omp else atc)*len(cfgs)*nruns for _,_,omp in METHODS)
    job=[0]
    def pr(label,cn,thr,r):
        job[0]+=1
        return '[%3d/%d] %-22s cfg=%s thr=%2d run=%d' % (job[0],total,label,cn,thr,r)
    for cfg in cfgs:
        cn,nx,ny,np_,mi = cfg
        gen_input(cfg)
        inf='input_%s.bin' % cn; sref='_sref_%s.out' % cn
        # serial
        stl=[]
        for r in range(nruns):
            tag=pr('serial',cn,1,r+1)
            print('  %s ...' % tag, end=' '); sys.stdout.flush()
            rc,out,_=run_cmd('./%s %s' % (bins['serial'],inf))
            t=parse_time(out)
            if t is not None: stl.append(t); print('%.4f s' % t)
            else: print('PARSE_FAIL')
        if os.path.exists('Mesh.out'): shutil.copy2('Mesh.out',sref)
        if stl:
            res.append(dict(method='serial',config=cn,NX=nx,NY=ny,num_points=np_,
                            threads=1,best_time=min(stl),avg_time=sum(stl)/len(stl),
                            all_times=list(stl)))
        # parallel
        for label,_,omp in PARALLEL_METHODS:
            b=bins[label]
            for thr in atc:
                tms=[]
                for r in range(nruns):
                    tag=pr(label,cn,thr,r+1)
                    print('  %s ...' % tag, end=' '); sys.stdout.flush()
                    rc,out,_=run_cmd('./%s %s' % (b,inf),env=make_env(thr))
                    t=parse_time(out)
                    if t is not None: tms.append(t); print('%.4f s' % t)
                    else: print('FAIL(rc=%d)' % rc)
                if os.path.exists('Mesh.out') and os.path.exists(sref):
                    ex,tok,md,nc=mesh_match('Mesh.out',sref)
                    cs='EXACT' if ex else ('TOL_OK' if tok else 'MISMATCH')
                else: cs='NO_REF'; md=-1.0
                xchk.append(dict(method=label,config=cn,threads=thr,status=cs,max_diff=md))
                if tms:
                    res.append(dict(method=label,config=cn,NX=nx,NY=ny,num_points=np_,
                                    threads=thr,best_time=min(tms),avg_time=sum(tms)/len(tms),
                                    all_times=list(tms)))
        if os.path.exists(sref): os.remove(sref)
        if os.path.exists(inf): os.remove(inf)
    print()
    return res, xchk

# -- Save CSVs --------------------------------------------------------
def save_all(res, xchk, ver, cfgs, tcounts):
    print('='*65 + '\nSAVING RESULTS\n' + '='*65)
    ensured(ODIR)
    ser = {}
    for r in res:
        if r['method']=='serial': ser[r['config']]=r['best_time']
    atc = sorted(set(r['threads'] for r in res if r['method']!='serial')) or [1]
    plabs = [m[0] for m in PARALLEL_METHODS]

    save_ver(ver)

    # cross_verification
    p=os.path.join(ODIR,'cross_verification.csv')
    with open(p,'w') as f:
        w=csv.writer(f); w.writerow(['method','config','threads','status','max_diff'])
        for c in xchk: w.writerow([c['method'],c['config'],c['threads'],c['status'],'%.2e'%c['max_diff']])
    print('  Written: %s' % p)

    # execution_times
    p=os.path.join(ODIR,'execution_times.csv')
    with open(p,'w') as f:
        w=csv.writer(f)
        mr=max(len(r['all_times']) for r in res) if res else 0
        hdr=['method','config','NX','NY','num_points','threads','best_time_s','avg_time_s','speedup_vs_serial']
        hdr+=['run%d_s'%(i+1) for i in range(mr)]
        w.writerow(hdr)
        for r in res:
            st=ser.get(r['config'],0.0); sp=st/r['best_time'] if st and r['best_time']>0 else 0
            row=[r['method'],r['config'],r['NX'],r['NY'],r['num_points'],r['threads'],
                 '%.6f'%r['best_time'],'%.6f'%r['avg_time'],'%.4f'%sp]
            row+=['%.6f'%t for t in r['all_times']]
            w.writerow(row)
    print('  Written: %s' % p)

    # speedup_table
    p=os.path.join(ODIR,'speedup_table.csv')
    with open(p,'w') as f:
        w=csv.writer(f)
        w.writerow(['config','NX','NY','num_points','method','threads',
                     'serial_time_s','parallel_time_s','speedup','efficiency_pct'])
        for r in res:
            if r['method']=='serial': continue
            st=ser.get(r['config'],0.0); sp=st/r['best_time'] if st and r['best_time']>0 else 0
            eff=sp/r['threads']*100 if r['threads']>0 else 0
            w.writerow([r['config'],r['NX'],r['NY'],r['num_points'],r['method'],r['threads'],
                        '%.6f'%st,'%.6f'%r['best_time'],'%.4f'%sp,'%.2f'%eff])
    print('  Written: %s' % p)

    # per-config tables
    for cn,nx,ny,np_,mi in cfgs:
        st=ser.get(cn,0.0)
        p=os.path.join(ODIR,'config_%s_times.csv'%cn)
        with open(p,'w') as f:
            w=csv.writer(f); w.writerow(['threads','serial']+plabs)
            for thr in atc:
                row=[thr,'%.6f'%st]
                for ml in plabs:
                    v=None
                    for r2 in res:
                        if r2['method']==ml and r2['config']==cn and r2['threads']==thr:
                            v=r2['best_time']; break
                    row.append('%.6f'%v if v else '')
                w.writerow(row)
        print('  Written: %s' % p)
        p=os.path.join(ODIR,'config_%s_speedup.csv'%cn)
        with open(p,'w') as f:
            w=csv.writer(f); w.writerow(['threads']+plabs)
            for thr in atc:
                row=[thr]
                for ml in plabs:
                    v=None
                    for r2 in res:
                        if r2['method']==ml and r2['config']==cn and r2['threads']==thr:
                            v=r2['best_time']; break
                    row.append('%.4f'%(st/v) if v and st and v>0 else '')
                w.writerow(row)
        print('  Written: %s' % p)

    # efficiency
    p=os.path.join(ODIR,'efficiency_table.csv')
    with open(p,'w') as f:
        w=csv.writer(f); w.writerow(['config','NX','NY','num_points','method','threads','efficiency_pct'])
        for r in res:
            if r['method']=='serial': continue
            st=ser.get(r['config'],0.0); sp=st/r['best_time'] if st and r['best_time']>0 else 0
            w.writerow([r['config'],r['NX'],r['NY'],r['num_points'],r['method'],r['threads'],
                        '%.2f'%(sp/r['threads']*100) if r['threads']>0 else '0'])
    print('  Written: %s' % p)

    # max_speedup_summary
    p=os.path.join(ODIR,'max_speedup_summary.csv')
    with open(p,'w') as f:
        w=csv.writer(f)
        w.writerow(['config','NX','NY','num_points','best_method','best_threads',
                     'serial_time_s','parallel_time_s','max_speedup','efficiency_pct'])
        for cn,nx,ny,np_,mi in cfgs:
            st=ser.get(cn,0.0); bsp=0.0; br=None
            for r in res:
                if r['method']=='serial' or r['config']!=cn: continue
                if st and r['best_time']>0:
                    sp=st/r['best_time']
                    if sp>bsp: bsp=sp; br=r
            if br:
                w.writerow([cn,nx,ny,np_,br['method'],br['threads'],
                            '%.6f'%st,'%.6f'%br['best_time'],'%.4f'%bsp,
                            '%.2f'%(bsp/br['threads']*100)])
    print('  Written: %s' % p)

    # summary
    print('\n'+'='*65+'\nSUMMARY\n'+'='*65)
    print('%-6s %-24s %10s %10s %4s %8s %7s' % ('Cfg','Method','Serial','Parallel','Thr','Speedup','Eff%'))
    print('-'*72)
    for cn,nx,ny,np_,mi in cfgs:
        st=ser.get(cn,0.0); bsp=0.0; br=None
        for r in res:
            if r['method']=='serial' or r['config']!=cn: continue
            if st and r['best_time']>0:
                sp=st/r['best_time']
                if sp>bsp: bsp=sp; br=r
        if br:
            print('%-6s %-24s %10.4f %10.4f %4d %7.2fx %6.1f%%' %
                  (cn,br['method'],st,br['best_time'],br['threads'],bsp,bsp/br['threads']*100))
    print()

# -- Main -------------------------------------------------------------
def main():
    quick = '--quick' in sys.argv
    if quick:
        print('*** QUICK MODE ***\n')
        cfgs=[c for c in CONFIGS if c[0] in ('a','c','e')]
        tc=[1,4,16]; nr=1
    else:
        cfgs=CONFIGS; tc=THREAD_COUNTS; nr=NUM_RUNS
    t0=time.time()
    check_files()
    bins=compile_all()
    ver=verify_test(bins,tc)
    print('='*65 + '\nGENERATING & BENCHMARKING INPUT FILES\n' + '='*65)
    res,xchk=do_benchmark(bins,cfgs,tc,nr)
    save_all(res,xchk,ver,cfgs,tc)
    el=time.time()-t0
    print('Wall time: %.1fs (%.1fmin)  |  Results in %s/' % (el,el/60,ODIR))

if __name__=='__main__':
    main()
