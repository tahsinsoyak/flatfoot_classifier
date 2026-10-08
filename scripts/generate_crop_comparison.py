import cv2
import numpy as np
import matplotlib.pyplot as plt

p_curr = cv2.imread('experiments/pes_planus_current.jpg')
p_stretch = cv2.imread('experiments/pes_planus_tight_stretch.jpg')
p_lb = cv2.imread('pes_padded.jpg')

n_curr = cv2.imread('experiments/normal_current.jpg')
n_stretch = cv2.imread('experiments/norm_sample_stretch.jpg')
n_lb = cv2.imread('experiments/norm_sample_letterbox.jpg')

fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
fig.suptitle('Röntgen Ön İşleme ve Kırpma Karşılaştırması (Pes Planus vs. Normal Ayak)', fontsize=16, fontweight='bold', y=0.98)
plt.subplots_adjust(wspace=0.08, hspace=0.15, left=0.08, right=0.96, top=0.88, bottom=0.04)

cols = [
    ('1. MEVCUT DURUM (Mevcut Kırpma)', 'Metal platform borusu + Uzun kaval kemiği + Harf etiketleri\nAyak tüm görselin yalnızca ~%35-40 alanını kaplıyor'),
    ('2. YENİ DURUM A (Tam Çerçeve / Full Crop)', 'Platform ve bacak kemikleri elendi, Yön sağa eşitlendi\nAyak %90 kaplar, yapay zeka için maksimum piksel verimliliği'),
    ('3. YENİ DURUM B (Doğal Orantı / Letterbox)', 'Platform ve bacak kemikleri elendi, Yön sağa eşitlendi\nKlinik anatomik açı ve en/boy oranını %100 korur')
]

rows = [
    ('DÜZ TABAN\n(Pes Planus)', [p_curr, p_stretch, p_lb], '#c0392b'),
    ('NORMAL AYAK\n(Normal)', [n_curr, n_stretch, n_lb], '#27ae60')
]

for r_idx, (r_label, imgs, color) in enumerate(rows):
    for c_idx, img in enumerate(imgs):
        ax = axes[r_idx, c_idx]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3.5 if c_idx > 0 else 1.5)
            
        if r_idx == 0:
            ax.set_title(f"{cols[c_idx][0]}\n{cols[c_idx][1]}", fontsize=11, fontweight='bold', pad=12)
            
    axes[r_idx, 0].set_ylabel(r_label, fontsize=13, fontweight='bold', color=color, labelpad=20)

out_path = 'experiments/crop_comparison_panel.png'
plt.savefig(out_path, bbox_inches='tight')
print('Updated', out_path)
