"""Generate publication-grade vector SVG architecture diagrams (English & Turkish) for the Flatfoot Classification Pipeline."""

from pathlib import Path


def create_architecture_svg(output_path: Path | str, lang: str = "en") -> str:
    is_tr = (lang == "tr")

    # Localized texts
    if is_tr:
        title = "AĞIRLIK AKTARMALI YAN AYAK GRAFİLERİNDEN DÜZ TABAN SINIFLANDIRMA PİPELİNE MİMARİSİ"
        subtitle = "Otomatik Tıbbi Ön İşleme • Kanonik Yön Hizalama • Derin Özellik Çıkarımı • Grad-CAM Klinik Açıklanabilirlik"
        
        st1_title = "AŞAMA 1: GRAFİ ALIMI & ÖN İŞLEME"
        st1_c1_title = "1.1 Klinik Yan Röntgen (16-bit)"
        st1_c1_p1 = "• 1.529 Ağırlık Aktarmalı Yan Ayak Grafisi"
        st1_c1_p2 = "• Yüksek dinamik aralık (0 – 65.535 uint16)"
        st1_c1_p3 = "• Değişken çözünürlük (~3000 × 2400 px)"
        st1_c1_p4 = "• Çift taraflı kohort (Sol ve Sağ ayaklar)"

        st1_c2_title = "1.2 Dinamik Yüzdelik Pencereleme"
        st1_c2_p1 = "• Yoğunluk kırpma: P₁ ve P₉₉ yüzdelikleri"
        st1_c2_p2 = "• Siyah kenarlık & aşırı ışın parlamalarını giderir"
        st1_c2_p3 = "✓ Normalize Edilmiş 8-bit Gri Düzey (0–255)"

        st1_c3_title = "1.3 CLAHE Kontrast İyileştirme"
        st1_c3_p1 = "• Kırpma sınırı = 2.0 (gürültü bastırma)"
        st1_c3_p2 = "• Izgara boyutu = 8 × 8 bağlamsal alt bölge"
        st1_c3_sub = "Belirginleşen Anatomik Yapılar:"
        st1_c3_a1 = "• Kalkaneus trabeküler kemik deseni"
        st1_c3_a2 = "• Talonaviküler eklem açıklığı"
        st1_c3_a3 = "• 1. Metatars-kuneiform dizilimi"
        st1_c3_badge = "İyileştirilmiş Mikro-Mimari"

        st2_title = "AŞAMA 2: MEKÂNSAL STANDARDIZASYON"
        st2_c1_title = "2.1 Tibia Ekseni Yön Tespiti"
        st2_c1_p1 = "• Üst %35–50 aralığında dikey kaval kemiği analizi"
        st2_c1_flip1 = "• X_tibia > W/2 (Sol ayak) → Yatay Aynalama"
        st2_c1_flip2 = "• X_tibia ≤ W/2 (Sağ ayak) → Koruma"
        st2_c1_badge = "✓ %100 Sağa Bakan Kanonik Ayak Arkı"

        st2_c2_title = "2.2 Basamak Tespiti & ROI Kırpma"
        st2_c2_p1 = "• Yatay Sobel (Ky) zemin ayrımı:"
        st2_c2_sub = "Sistematik Artefakt Temizliği:"
        st2_c2_a1 = "✖ Alt kısımdaki metal basamak & aparatlar"
        st2_c2_a2 = "✖ Üst bacak kemikleri & 'L'/'R' harf etiketleri"
        st2_c2_badge = "✓ İzole Edilmiş Anatomik Ayak Kompleksi"

        st2_c3_title = "2.3 Kanonik 512×512 Yeniden Boyutlandırma"
        st2_c3_p1 = "• Yüksek doğruluklu çift doğrusal enterpolasyon"
        st2_c3_p2 = "• Trabeküler kemik frekanslarını korur"
        st2_c3_p3 = "• Standart girdi tensörü: [3 × 512 × 512]"
        st2_c3_badge = "Homojen Model Girdi Uzayı"

        st3_title = "AŞAMA 3: DERİN ÖZELLİK OMURGASI"
        st3_c1_title = "3.1 Karşılaştırılan Omurga Modelleri"
        st3_c1_m1 = "EfficientNet-B2 (En Yüksek Genel Başarım)"
        st3_c1_m1_sub = "• Bileşik ölçekleme • Squeeze-and-Excitation • 9.2M param"
        st3_c1_m2 = "ResNet-50 (En Yüksek Duyarlılık)"
        st3_c1_m2_sub = "• Rezidüel artık bloklar • 25.6M param"
        st3_c1_m3 = "ConvNeXt-Tiny (Modern Konvolüsyonel Ağ)"
        st3_c1_m3_sub = "• 7×7 derinlik konvolüsyonu • Ters darboğaz • 28.6M"
        st3_c1_custom = "+ Özgün FootArchNet Mimari Tabanı"

        st3_c2_title = "3.2 Eğitim ve Optimizasyon Düzeni"
        st3_c2_p1 = "• Donanım: NVIDIA RTX 3050 Ti (4GB VRAM)"
        st3_c2_p2 = "• AMP: Otomatik Karışık Hassasiyet (FP16)"
        st3_c2_p3 = "• Optimize Edici: AdamW (LR=1e-4, Decay=1e-2)"
        st3_c2_p4 = "• Takvim: Kosinüs Tavlama (20 Epok)"
        st3_c2_p5 = "• Yığın boyutu: 8 (Kararlı 1.900 MB VRAM profili)"
        st3_c2_badge = "Sıfır Veri Sızıntılı Tabakalı Bölümleme"

        st4_title = "AŞAMA 4: TANI VE AÇIKLANABİLİRLİK (XAI)"
        st4_c1_title = "4.1 Klinik Test Değerlendirmesi (n=230)"
        st4_c1_top = "Lider Model: EfficientNet-B2"
        st4_c1_acc = "• Doğruluk (Accuracy):"
        st4_c1_auc = "• ROC-AUC Skoru:"
        st4_c1_sens = "• Duyarlılık (Recall):"
        st4_c1_sens_sub = "(ResNet: %94.89)"
        st4_c1_spec = "• Özgüllük (Specificity):"
        st4_c1_f1 = "• F1-Skor:"
        st4_c1_note = "• 137 düz taban vakasından sadece 7–8'i kaçırıldı"
        st4_c1_badge = "✓ Mükemmel Birinci Basamak Tarama Gücü"

        st4_c2_title = "4.2 Grad-CAM Görsel Açıklanabilirlik"
        st4_c2_p1 = "• Sınıf ayırt edici gradyan haritalandırması:"
        st4_c2_sub = "Fizyolojik Olarak Doğrulanmış Odak Noktaları:"
        st4_c2_a1 = "🔴 Medial Longitudinal Ark Çökmesi"
        st4_c2_a2 = "🟡 Naviküler-Kuneiform Düşüşü"
        st4_c2_a3 = "🟢 Kalkaneal Eğim (Pitch) Açılanması"
        st4_c2_sc1 = "Sıfır Kestirme Öğrenme (No Shortcut)"
        st4_c2_sc2 = "Metal basamak, metin etiketleri ve boşluğu yok sayar"
    else:
        title = "END-TO-END DEEP LEARNING FRAMEWORK FOR RADIOGRAPHIC PES PLANUS CLASSIFICATION"
        subtitle = "Automated Medical Preprocessing • Canonical Alignment • Deep Feature Extraction • Grad-CAM Clinical Interpretability"

        st1_title = "STAGE 1: RADIOGRAPH INGESTION"
        st1_c1_title = "1.1 Clinical Lateral X-Ray (16-bit)"
        st1_c1_p1 = "• 1,529 Weight-Bearing Radiographs"
        st1_c1_p2 = "• High-dynamic range (0 – 65,535 uint16)"
        st1_c1_p3 = "• Variable resolution (~3000 × 2400 px)"
        st1_c1_p4 = "• Bilateral cohort (both Left & Right feet)"

        st1_c2_title = "1.2 Dynamic Percentile Windowing"
        st1_c2_p1 = "• Intensity clipping between P₁ & P₉₉"
        st1_c2_p2 = "• Removes black border / bright beam spikes"
        st1_c2_p3 = "✓ Normalized 8-bit Grayscale (0–255)"

        st1_c3_title = "1.3 CLAHE Contrast Enhancement"
        st1_c3_p1 = "• Clip limit = 2.0 (suppresses noise)"
        st1_c3_p2 = "• Tile grid = 8 × 8 contextual sub-regions"
        st1_c3_sub = "Sharpened Anatomical Details:"
        st1_c3_a1 = "• Calcaneal trabeculae & cortex"
        st1_c3_a2 = "• Talonavicular joint margin"
        st1_c3_a3 = "• 1st Metatarsal-cuneiform alignment"
        st1_c3_badge = "Enhanced Micro-Architecture"

        st2_title = "STAGE 2: SPATIAL STANDARDIZATION"
        st2_c1_title = "2.1 Tibial-Axis Laterality Detection"
        st2_c1_p1 = "• Analyzes vertical shaft in upper 35–50%"
        st2_c1_flip1 = "• If X_tibia > W/2 (Left foot) → Horizontal Flip"
        st2_c1_flip2 = "• If X_tibia ≤ W/2 (Right foot) → Preserve"
        st2_c1_badge = "✓ 100% Right-Facing Canonical Arch"

        st2_c2_title = "2.2 Platform Edge & ROI Cropping"
        st2_c2_p1 = "• Horizontal Sobel filter (Ky) detects ground:"
        st2_c2_sub = "Systematic Artifact Elimination:"
        st2_c2_a1 = "✖ Metal table base & floor bars below"
        st2_c2_a2 = "✖ Upper tibia shaft above & 'L'/'R' tags"
        st2_c2_badge = "✓ Isolated Anatomical Foot Complex"

        st2_c3_title = "2.3 Canonical 512×512 Resizing"
        st2_c3_p1 = "• High-fidelity bilinear interpolation"
        st2_c3_p2 = "• Preserves trabecular bone frequencies"
        st2_c3_p3 = "• Standardized tensor: [3 × 512 × 512]"
        st2_c3_badge = "Uniform Model Input Space"

        st3_title = "STAGE 3: DEEP FEATURE BACKBONE"
        st3_c1_title = "3.1 Benchmarked Backbone Networks"
        st3_c1_m1 = "EfficientNet-B2 (Top Overall)"
        st3_c1_m1_sub = "• Compound scaling • Squeeze-and-Excitation • 9.2M params"
        st3_c1_m2 = "ResNet-50 (Top Sensitivity)"
        st3_c1_m2_sub = "• Bottleneck Residual Blocks • 25.6M params"
        st3_c1_m3 = "ConvNeXt-Tiny (Modern ConvNet)"
        st3_c1_m3_sub = "• 7×7 Depthwise Convolutions • Inverted Bottleneck • 28.6M"
        st3_c1_custom = "+ Custom FootArchNet Architecture Base"

        st3_c2_title = "3.2 Training & Optimization Scheme"
        st3_c2_p1 = "• Hardware: NVIDIA RTX 3050 Ti (4GB VRAM)"
        st3_c2_p2 = "• AMP: Automatic Mixed Precision (FP16)"
        st3_c2_p3 = "• Optimizer: AdamW (LR=1e-4, Decay=1e-2)"
        st3_c2_p4 = "• Schedule: Cosine Annealing (20 Epochs)"
        st3_c2_p5 = "• Batch size: 8 (Stable 1,900 MB VRAM profile)"
        st3_c2_badge = "Zero Data Leakage Stratified Split"

        st4_title = "STAGE 4: DIAGNOSIS & XAI"
        st4_c1_title = "4.1 Clinical Test Evaluation (n=230)"
        st4_c1_top = "Top Benchmark: EfficientNet-B2"
        st4_c1_acc = "• Accuracy:"
        st4_c1_auc = "• ROC-AUC:"
        st4_c1_sens = "• Sensitivity:"
        st4_c1_sens_sub = "(ResNet: 94.89%)"
        st4_c1_spec = "• Specificity:"
        st4_c1_f1 = "• F1-Score:"
        st4_c1_note = "• Missing only 7–8 out of 137 flatfoot cases"
        st4_c1_badge = "✓ Excellent First-Line Diagnostic Screen"

        st4_c2_title = "4.2 Grad-CAM Visual Explainability"
        st4_c2_p1 = "• Class-discriminative gradient mapping:"
        st4_c2_sub = "Physiologically Validated Saliency:"
        st4_c2_a1 = "🔴 Medial Longitudinal Arch Collapse"
        st4_c2_a2 = "🟡 Navicular-Cuneiform Plantar Drop"
        st4_c2_a3 = "🟢 Calcaneal Pitch Angle Inclination"
        st4_c2_sc1 = "Zero Shortcut Learning"
        st4_c2_sc2 = "Ignores text tags, borders, and metal stands"

    acc_val = "%86.09" if is_tr else "86.09%"
    sens_val = "%91.24" if is_tr else "91.24%"
    spec_val = "%78.49" if is_tr else "78.49%"

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 680" width="100%" height="100%" style="background:#ffffff; font-family:'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <!-- Gradients -->
    <linearGradient id="gradHeader" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e3a8a" />
    </linearGradient>
    <linearGradient id="gradStage1" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f0f9ff" />
      <stop offset="100%" stop-color="#e0f2fe" />
    </linearGradient>
    <linearGradient id="gradStage2" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f5f3ff" />
      <stop offset="100%" stop-color="#ede9fe" />
    </linearGradient>
    <linearGradient id="gradStage3" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#ecfdf5" />
      <stop offset="100%" stop-color="#d1fae5" />
    </linearGradient>
    <linearGradient id="gradStage4" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fffbeb" />
      <stop offset="100%" stop-color="#fef3c7" />
    </linearGradient>

    <!-- Drop Shadows -->
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.08" />
    </filter>
    <filter id="cardShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.06" />
    </filter>

    <!-- Arrow Markers -->
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#0284c7" />
    </marker>
    <marker id="arrowPurple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#6d28d9" />
    </marker>
    <marker id="arrowGreen" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669" />
    </marker>
    <marker id="arrowAmber" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#d97706" />
    </marker>
  </defs>

  <!-- Title & Banner -->
  <rect x="25" y="18" width="1150" height="56" rx="8" fill="url(#gradHeader)" filter="url(#shadow)" />
  <text x="600" y="44" fill="#ffffff" font-size="16" font-weight="700" text-anchor="middle" letter-spacing="0.4">
    {title}
  </text>
  <text x="600" y="63" fill="#93c5fd" font-size="11.5" text-anchor="middle">
    {subtitle}
  </text>

  <!-- ================= STAGE 1: INGESTION & WINDOWING ================= -->
  <g transform="translate(25, 92)">
    <rect x="0" y="0" width="265" height="560" rx="10" fill="url(#gradStage1)" stroke="#bae6fd" stroke-width="1.5" filter="url(#shadow)" />
    
    <!-- Stage Header -->
    <rect x="12" y="12" width="241" height="34" rx="6" fill="#0284c7" />
    <text x="132" y="34" fill="#ffffff" font-size="11.5" font-weight="700" text-anchor="middle">
      {st1_title}
    </text>

    <!-- Card 1: Raw DICOM -->
    <g transform="translate(14, 58)">
      <rect x="0" y="0" width="237" height="105" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" filter="url(#cardShadow)" />
      <text x="12" y="22" fill="#0369a1" font-size="11" font-weight="700">{st1_c1_title}</text>
      <text x="12" y="42" fill="#475569" font-size="9.5">{st1_c1_p1}</text>
      <text x="12" y="58" fill="#475569" font-size="9.5">{st1_c1_p2}</text>
      <text x="12" y="74" fill="#475569" font-size="9.5">{st1_c1_p3}</text>
      <text x="12" y="90" fill="#64748b" font-size="9" font-style="italic">{st1_c1_p4}</text>
    </g>

    <!-- Down Arrow 1 -->
    <path d="M 132 170 L 132 185" stroke="#0284c7" stroke-width="2" marker-end="url(#arrow)" />

    <!-- Card 2: Dynamic Windowing -->
    <g transform="translate(14, 192)">
      <rect x="0" y="0" width="237" height="150" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" filter="url(#cardShadow)" />
      <text x="12" y="22" fill="#0369a1" font-size="11" font-weight="700">{st1_c2_title}</text>
      <text x="12" y="40" fill="#475569" font-size="9.5">{st1_c2_p1}</text>
      
      <!-- Formula Box -->
      <rect x="10" y="50" width="217" height="48" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
      <text x="118" y="70" fill="#0f172a" font-size="10.5" font-family="'Times New Roman', serif" font-style="italic" text-anchor="middle">
        I<tspan font-size="8" dy="2">8-bit</tspan><tspan dy="-2"> = clip( (I</tspan><tspan font-size="8" dy="2">16-bit</tspan><tspan dy="-2"> - P₁) / ΔP , 0, 1) × 255</tspan>
      </text>
      <text x="118" y="88" fill="#64748b" font-size="8.5" text-anchor="middle">
        ΔP = P₉₉ - P₁ + 10⁻⁶
      </text>

      <text x="12" y="116" fill="#475569" font-size="9.5">{st1_c2_p2}</text>
      <text x="12" y="132" fill="#059669" font-size="9.5" font-weight="600">{st1_c2_p3}</text>
    </g>

    <!-- Down Arrow 2 -->
    <path d="M 132 349 L 132 364" stroke="#0284c7" stroke-width="2" marker-end="url(#arrow)" />

    <!-- Card 3: CLAHE Contrast -->
    <g transform="translate(14, 371)">
      <rect x="0" y="0" width="237" height="168" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" filter="url(#cardShadow)" />
      <text x="12" y="22" fill="#0369a1" font-size="11" font-weight="700">{st1_c3_title}</text>
      <text x="12" y="42" fill="#475569" font-size="9.5">{st1_c3_p1}</text>
      <text x="12" y="58" fill="#475569" font-size="9.5">{st1_c3_p2}</text>
      <text x="12" y="76" fill="#0369a1" font-size="9.5" font-weight="600">{st1_c3_sub}</text>
      <text x="20" y="94" fill="#475569" font-size="9">{st1_c3_a1}</text>
      <text x="20" y="110" fill="#475569" font-size="9">{st1_c3_a2}</text>
      <text x="20" y="126" fill="#475569" font-size="9">{st1_c3_a3}</text>
      <rect x="12" y="138" width="213" height="20" rx="3" fill="#e0f2fe" />
      <text x="118" y="152" fill="#0284c7" font-size="9" font-weight="700" text-anchor="middle">{st1_c3_badge}</text>
    </g>
  </g>

  <!-- Arrow: Stage 1 -> Stage 2 -->
  <path d="M 290 372 L 315 372" stroke="#0284c7" stroke-width="2.5" marker-end="url(#arrow)" />

  <!-- ================= STAGE 2: CANONICAL STANDARDIZATION ================= -->
  <g transform="translate(320, 92)">
    <rect x="0" y="0" width="265" height="560" rx="10" fill="url(#gradStage2)" stroke="#ddd6fe" stroke-width="1.5" filter="url(#shadow)" />
    
    <!-- Stage Header -->
    <rect x="12" y="12" width="241" height="34" rx="6" fill="#6d28d9" />
    <text x="132" y="34" fill="#ffffff" font-size="11.5" font-weight="700" text-anchor="middle">
      {st2_title}
    </text>

    <!-- Card 1: Tibial Shaft Axis -->
    <g transform="translate(14, 58)">
      <rect x="0" y="0" width="237" height="152" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" filter="url(#cardShadow)" />
      <text x="12" y="22" fill="#5b21b6" font-size="11" font-weight="700">{st2_c1_title}</text>
      <text x="12" y="40" fill="#475569" font-size="9.5">{st2_c1_p1}</text>

      <!-- Density formula -->
      <rect x="10" y="48" width="217" height="48" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
      <text x="118" y="68" fill="#0f172a" font-size="10.5" font-family="'Times New Roman', serif" font-style="italic" text-anchor="middle">
        ρ<tspan font-size="8" dy="2">vert</tspan><tspan dy="-2">(x) = </tspan><tspan font-size="13">∑</tspan><tspan font-size="8" dy="3">y=0.35H..0.50H</tspan><tspan font-size="10.5" dy="-3"> I(x, y)</tspan>
      </text>
      <text x="118" y="87" fill="#64748b" font-size="8.5" text-anchor="middle">
        X<tspan font-size="7">tibia</tspan> = argmax (ρ<tspan font-size="7">vert</tspan> * G<tspan font-size="7">σ=15</tspan>)
      </text>

      <text x="12" y="114" fill="#475569" font-size="9.5">{st2_c1_flip1}</text>
      <text x="12" y="130" fill="#475569" font-size="9.5">{st2_c1_flip2}</text>
      <text x="12" y="145" fill="#5b21b6" font-size="9" font-weight="700">{st2_c1_badge}</text>
    </g>

    <!-- Down Arrow 1 -->
    <path d="M 132 217 L 132 232" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrowPurple)" />

    <!-- Card 2: Platform Edge & ROI -->
    <g transform="translate(14, 239)">
      <rect x="0" y="0" width="237" height="150" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" filter="url(#cardShadow)" />
      <text x="12" y="22" fill="#5b21b6" font-size="11" font-weight="700">{st2_c2_title}</text>
      <text x="12" y="40" fill="#475569" font-size="9.5">{st2_c2_p1}</text>

      <rect x="10" y="48" width="217" height="34" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
      <text x="118" y="70" fill="#0f172a" font-size="10" font-family="'Times New Roman', serif" font-style="italic" text-anchor="middle">
        Y<tspan font-size="7">platform</tspan> = argmax<tspan font-size="7" dy="2">y∈[0.55..0.95H]</tspan><tspan dy="-2"> </tspan><tspan font-size="12">∑</tspan><tspan font-size="10"> |S<tspan font-size="7">y</tspan>(x, y)|</tspan>
      </text>

      <text x="12" y="100" fill="#475569" font-size="9.5">{st2_c2_sub}</text>
      <text x="20" y="116" fill="#b91c1c" font-size="9">{st2_c2_a1}</text>
      <text x="20" y="130" fill="#b91c1c" font-size="9">{st2_c2_a2}</text>
      <text x="12" y="145" fill="#059669" font-size="9" font-weight="700">{st2_c2_badge}</text>
    </g>

    <!-- Down Arrow 2 -->
    <path d="M 132 396 L 132 411" stroke="#6d28d9" stroke-width="2" marker-end="url(#arrowPurple)" />

    <!-- Card 3: Uniform Dimension -->
    <g transform="translate(14, 418)">
      <rect x="0" y="0" width="237" height="121" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" filter="url(#cardShadow)" />
      <text x="12" y="22" fill="#5b21b6" font-size="11" font-weight="700">{st2_c3_title}</text>
      <text x="12" y="42" fill="#475569" font-size="9.5">{st2_c3_p1}</text>
      <text x="12" y="58" fill="#475569" font-size="9.5">{st2_c3_p2}</text>
      <text x="12" y="74" fill="#475569" font-size="9.5">{st2_c3_p3}</text>
      <rect x="12" y="88" width="213" height="22" rx="3" fill="#ede9fe" />
      <text x="118" y="103" fill="#6d28d9" font-size="9.5" font-weight="700" text-anchor="middle">
        {st2_c3_badge}
      </text>
    </g>
  </g>

  <!-- Arrow: Stage 2 -> Stage 3 -->
  <path d="M 585 372 L 610 372" stroke="#6d28d9" stroke-width="2.5" marker-end="url(#arrowPurple)" />

  <!-- ================= STAGE 3: DEEP CONVOLUTIONAL BACKBONE ================= -->
  <g transform="translate(615, 92)">
    <rect x="0" y="0" width="275" height="560" rx="10" fill="url(#gradStage3)" stroke="#a7f3d0" stroke-width="1.5" filter="url(#shadow)" />
    
    <!-- Stage Header -->
    <rect x="12" y="12" width="251" height="34" rx="6" fill="#059669" />
    <text x="137" y="34" fill="#ffffff" font-size="11.5" font-weight="700" text-anchor="middle">
      {st3_title}
    </text>

    <!-- CNN Architecture Card -->
    <g transform="translate(14, 58)">
      <rect x="0" y="0" width="247" height="235" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" filter="url(#cardShadow)" />
      <text x="12" y="22" fill="#047857" font-size="11" font-weight="700">{st3_c1_title}</text>
      
      <!-- Network 1 -->
      <rect x="10" y="34" width="227" height="48" rx="4" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1" />
      <text x="18" y="52" fill="#065f46" font-size="10" font-weight="700">{st3_c1_m1}</text>
      <text x="18" y="68" fill="#475569" font-size="8.5">{st3_c1_m1_sub}</text>

      <!-- Network 2 -->
      <rect x="10" y="88" width="227" height="48" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
      <text x="18" y="106" fill="#1e293b" font-size="10" font-weight="700">{st3_c1_m2}</text>
      <text x="18" y="122" fill="#475569" font-size="8.5">{st3_c1_m2_sub}</text>

      <!-- Network 3 -->
      <rect x="10" y="142" width="227" height="48" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
      <text x="18" y="160" fill="#1e293b" font-size="10" font-weight="700">{st3_c1_m3}</text>
      <text x="18" y="176" fill="#475569" font-size="8.5">{st3_c1_m3_sub}</text>

      <!-- Custom Future Badge -->
      <rect x="10" y="196" width="227" height="28" rx="4" fill="#ecfdf5" stroke="#34d399" stroke-width="1" />
      <text x="123" y="214" fill="#059669" font-size="9" font-weight="700" text-anchor="middle">
        {st3_c1_custom}
      </text>
    </g>

    <!-- Down Arrow 1 -->
    <path d="M 137 300 L 137 315" stroke="#059669" stroke-width="2" marker-end="url(#arrowGreen)" />

    <!-- Optimization Card -->
    <g transform="translate(14, 322)">
      <rect x="0" y="0" width="247" height="217" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" filter="url(#cardShadow)" />
      <text x="12" y="22" fill="#047857" font-size="11" font-weight="700">{st3_c2_title}</text>
      
      <!-- Loss Equation -->
      <rect x="10" y="32" width="227" height="52" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
      <text x="123" y="52" fill="#0f172a" font-size="10" font-family="'Times New Roman', serif" font-style="italic" text-anchor="middle">
        L<tspan font-size="7">CE</tspan> = - <tspan font-size="11">∑</tspan> w<tspan font-size="7">c</tspan> [ y log(p̂) + (1-y) log(1-p̂) ]
      </text>
      <text x="123" y="72" fill="#64748b" font-size="8.5" text-anchor="middle">
        w<tspan font-size="7">normal</tspan> = 1.231,  w<tspan font-size="7">pes_planus</tspan> = 0.842
      </text>

      <text x="12" y="102" fill="#475569" font-size="9.5">{st3_c2_p1}</text>
      <text x="12" y="118" fill="#475569" font-size="9.5">{st3_c2_p2}</text>
      <text x="12" y="134" fill="#475569" font-size="9.5">{st3_c2_p3}</text>
      <text x="12" y="150" fill="#475569" font-size="9.5">{st3_c2_p4}</text>
      <text x="12" y="166" fill="#475569" font-size="9.5">{st3_c2_p5}</text>
      
      <rect x="10" y="178" width="227" height="28" rx="3" fill="#d1fae5" />
      <text x="123" y="196" fill="#065f46" font-size="9.5" font-weight="700" text-anchor="middle">
        {st3_c2_badge}
      </text>
    </g>
  </g>

  <!-- Arrow: Stage 3 -> Stage 4 -->
  <path d="M 890 372 L 915 372" stroke="#059669" stroke-width="2.5" marker-end="url(#arrowGreen)" />

  <!-- ================= STAGE 4: CLINICAL DIAGNOSIS & EXPLAINABILITY ================= -->
  <g transform="translate(920, 92)">
    <rect x="0" y="0" width="255" height="560" rx="10" fill="url(#gradStage4)" stroke="#fde68a" stroke-width="1.5" filter="url(#shadow)" />
    
    <!-- Stage Header -->
    <rect x="12" y="12" width="231" height="34" rx="6" fill="#d97706" />
    <text x="127" y="34" fill="#ffffff" font-size="11.5" font-weight="700" text-anchor="middle">
      {st4_title}
    </text>

    <!-- Card 1: Benchmark Metrics -->
    <g transform="translate(14, 58)">
      <rect x="0" y="0" width="227" height="185" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" filter="url(#cardShadow)" />
      <text x="12" y="22" fill="#b45309" font-size="11" font-weight="700">{st4_c1_title}</text>

      <!-- Mini Table -->
      <rect x="8" y="32" width="211" height="110" rx="4" fill="#fffbeb" stroke="#fef3c7" stroke-width="1" />
      <text x="16" y="50" fill="#92400e" font-size="9" font-weight="700">{st4_c1_top}</text>
      <text x="16" y="68" fill="#1e293b" font-size="9.5">{st4_c1_acc} <tspan font-weight="700" fill="#047857">{acc_val}</tspan></text>
      <text x="16" y="84" fill="#1e293b" font-size="9.5">{st4_c1_auc} <tspan font-weight="700" fill="#047857">0.9222</tspan></text>
      <text x="16" y="100" fill="#1e293b" font-size="9.5">{st4_c1_sens} <tspan font-weight="700" fill="#047857">{sens_val}</tspan> <tspan font-size="8" fill="#64748b">{st4_c1_sens_sub}</tspan></text>
      <text x="16" y="116" fill="#1e293b" font-size="9.5">{st4_c1_spec} <tspan font-weight="700" fill="#047857">{spec_val}</tspan></text>
      <text x="16" y="132" fill="#1e293b" font-size="9.5">{st4_c1_f1} <tspan font-weight="700" fill="#047857">0.8865</tspan></text>

      <text x="12" y="158" fill="#475569" font-size="8.5">{st4_c1_note}</text>
      <text x="12" y="174" fill="#059669" font-size="9" font-weight="700">{st4_c1_badge}</text>
    </g>

    <!-- Down Arrow -->
    <path d="M 127 250 L 127 265" stroke="#d97706" stroke-width="2" marker-end="url(#arrowAmber)" />

    <!-- Card 2: Grad-CAM Explainability -->
    <g transform="translate(14, 272)">
      <rect x="0" y="0" width="227" height="267" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" filter="url(#cardShadow)" />
      <text x="12" y="22" fill="#b45309" font-size="11" font-weight="700">{st4_c2_title}</text>
      <text x="12" y="38" fill="#475569" font-size="9">{st4_c2_p1}</text>

      <!-- Grad-CAM formula -->
      <rect x="8" y="46" width="211" height="58" rx="4" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
      <text x="113" y="66" fill="#0f172a" font-size="9.5" font-family="'Times New Roman', serif" font-style="italic" text-anchor="middle">
        α<tspan font-size="7">k</tspan><tspan font-size="7">c</tspan> = (1/Z) <tspan font-size="12">∑</tspan><tspan font-size="7" dy="2">i</tspan><tspan font-size="12" dy="-2">∑</tspan><tspan font-size="7" dy="2">j</tspan><tspan dy="-2"> (∂y</tspan><tspan font-size="7">c</tspan> / ∂A<tspan font-size="7">ij</tspan><tspan font-size="7">k</tspan>)
      </text>
      <text x="113" y="88" fill="#0f172a" font-size="10" font-family="'Times New Roman', serif" font-style="italic" text-anchor="middle">
        L<tspan font-size="7">Grad-CAM</tspan><tspan font-size="7">c</tspan> = ReLU( <tspan font-size="11">∑</tspan><tspan font-size="7">k</tspan> α<tspan font-size="7">k</tspan><tspan font-size="7">c</tspan> A<tspan font-size="7">k</tspan> )
      </text>

      <text x="12" y="122" fill="#0369a1" font-size="9.5" font-weight="700">{st4_c2_sub}</text>
      
      <rect x="8" y="132" width="211" height="24" rx="3" fill="#fef2f2" stroke="#fecaca" stroke-width="0.8" />
      <text x="16" y="148" fill="#991b1b" font-size="8.5" font-weight="600">{st4_c2_a1}</text>

      <rect x="8" y="160" width="211" height="24" rx="3" fill="#fffbeb" stroke="#fde68a" stroke-width="0.8" />
      <text x="16" y="176" fill="#92400e" font-size="8.5" font-weight="600">{st4_c2_a2}</text>

      <rect x="8" y="188" width="211" height="24" rx="3" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="0.8" />
      <text x="16" y="204" fill="#166534" font-size="8.5" font-weight="600">{st4_c2_a3}</text>

      <rect x="8" y="218" width="211" height="38" rx="4" fill="#1e293b" />
      <text x="113" y="233" fill="#f8fafc" font-size="8.5" font-weight="600" text-anchor="middle">
        {st4_c2_sc1}
      </text>
      <text x="113" y="247" fill="#94a3b8" font-size="7.5" text-anchor="middle">
        {st4_c2_sc2}
      </text>
    </g>
  </g>
</svg>'''
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return svg


if __name__ == "__main__":
    p_en = Path(__file__).resolve().parent.parent / "papers" / "methodology_architecture_en.svg"
    p_tr = Path(__file__).resolve().parent.parent / "papers" / "methodology_architecture_tr.svg"
    create_architecture_svg(p_en, "en")
    create_architecture_svg(p_tr, "tr")
    print(f"Generated {p_en} ({p_en.stat().st_size} bytes)")
    print(f"Generated {p_tr} ({p_tr.stat().st_size} bytes)")
