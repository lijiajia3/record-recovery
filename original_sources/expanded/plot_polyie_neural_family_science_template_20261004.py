"""Plot only the complete hash-verified neural-family report; no model/Gold reads."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from polyie_local_adapter import ROOT,sha
from polyie_neural_family import DEST_REL,FREEZE_REL


def main():
    path=ROOT/DEST_REL/'test_summary.json';report=json.loads(path.read_text())
    assert report['complete_graph_barrier']==9 and report['test_gold_loaded_only_after_all_nine_source_graphs']
    assert report['freeze_sha256']==sha(ROOT/FREEZE_REL)
    assert report['generation_manifest_sha256']==sha(ROOT/DEST_REL/'generation_manifest.json')
    results=report['architecture_results'];order=['mean','capacity_mean','typed']
    labels=['Mean + linear','Mean + matched head','Role + position head']
    colors=['#737B86','#416E99','#C25D30']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(6.5,6.3))
    grid=fig.add_gridspec(2,2,height_ratios=[1,1],width_ratios=[1,1.18])
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
    ax=axes[0]
    for i,name in enumerate(order):
        r=results[name];f1=100*r['pooled_primary']['f1'];low,high=np.asarray(r['paper_bootstrap_95_f1_ci'])*100
        seeds=[100*s['covered_primary']['total']['f1']for s in r['all_seeds'].values()]
        ax.scatter(np.full(3,i)+[-.1,0,.1],seeds,s=18,facecolors='none',edgecolors=colors[i],linewidths=.8,zorder=2)
        ax.vlines(i,low,high,color=colors[i],lw=2,zorder=3)
        ax.plot(i,f1,'s',color=colors[i],markersize=5,zorder=4)
    ax.set_xticks(range(3),['Mean\nlinear','Mean\nmatched','Role +\nposition']);ax.set_ylabel('Complete-group micro-F1 (%)')
    ax.set_ylim(bottom=0);ax.set_title('A   Held-out extraction',loc='left',fontweight='bold')
    ax.text(0,-.34,'Squares: pooled counts; circles: fitted seeds\nLines: 95% paper-bootstrap intervals',transform=ax.transAxes,fontsize=7.5)
    ax=axes[1];ax.axvline(0,color='#A0A0A0',lw=.8,ls='--')
    contrasts=report['planned_contrasts']
    for i,c in enumerate(contrasts):
        lo,hi=c['paired_paper_bootstrap_ci95'];ax.hlines(i,lo,hi,color='#C25D30',lw=2)
        ax.plot(c['delta_f1_points'],i,'s',color='#C25D30',ms=5)
        ax.annotate('Holm p = '+format(c['holm_p_two_planned_neural_contrasts'],'.3g'),
            (c['delta_f1_points'],i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=7.5)
    ax.set_yticks([0,1],['vs mean\nlinear','vs mean\nmatched']);ax.set_ylim(-.6,1.65)
    ax.set_xlabel('Role/position minus control (F1 points)');ax.set_title('B   Paired contrasts',loc='left',fontweight='bold')
    ax=axes[2];papers=report['paper_ids']
    def f1(c):return 100*2*c['tp']/(2*c['tp']+c['fp']+c['fn'])if 2*c['tp']+c['fp']+c['fn']else 0
    differences=[f1(results['typed']['pooled_by_original_paper'][p])-f1(results['capacity_mean']['pooled_by_original_paper'][p])for p in papers]
    ax.bar(range(len(papers)),differences,color=['#C25D30'if d>=0 else'#416E99'for d in differences],width=.7)
    ax.axhline(0,color='#A0A0A0',lw=.8);ax.set_xticks(range(len(papers)),papers,rotation=90,fontsize=7.5)
    ax.set_xlabel('Original source paper');ax.set_ylabel('Matched-control difference (F1 points)')
    ax.set_title('C   Paper heterogeneity',loc='left',fontweight='bold')
    fig.subplots_adjust(left=.11,right=.985,bottom=.20,top=.92,wspace=.62,hspace=.95)
    dest=ROOT/'figures';files=[]
    assert not any(dest.glob('figure4_neural_ablation_science_template_20261004*')), 'Preserve previous original render'
    for extension in ['png','pdf','svg']:
        p=dest/('figure4_neural_ablation_science_template_20261004.'+extension);fig.savefig(p,dpi=300);files.append(p)
    plt.close(fig)
    source=dest/'figure4_neural_ablation_science_template_20261004_source.csv'
    with source.open('w',newline='')as stream:
        writer=csv.writer(stream);writer.writerow(['architecture','seed_or_pooled','paper','tp','fp','fn','f1'])
        for name in order:
            for seed,r in results[name]['all_seeds'].items():
                for paper,c in r['covered_primary']['by_paper'].items():writer.writerow([name,seed,paper,c['tp'],c['fp'],c['fn'],f1(c)/100])
            for paper,c in results[name]['pooled_by_original_paper'].items():writer.writerow([name,'pooled',paper,c['tp'],c['fp'],c['fn'],f1(c)/100])
    files.append(source)
    (dest/'figure4_neural_ablation_science_template_20261004_manifest.json').write_text(json.dumps({'summary_sha256':sha(path),
        'files':{str(p.relative_to(ROOT)):sha(p)for p in files},'bootstrap_unit':'original paper',
        'seeds_are_independent_papers':False,'new_tests_created_by_figure':False},indent=2)+'\n')


if __name__=='__main__':main()
