/* Opțiuni globale */
options nodate nonumber pagesize=60 linesize=120 validvarname=v7;
ods graphics on / width=800px height=500px;

/* Macro variabilă: calea la fișierele de date */
%let data_path = /home/u64479897/Hidroelectrica/date;

/* === FUNCTIA 1 — Import CSV-uri === */
proc import datafile="&data_path./hidroelectrica_consolidat_2021_2025.csv"
    out=work.financiar
    dbms=csv replace;
    guessingrows=100;
    getnames=yes;
run;

proc import datafile="&data_path./hidroelectrica_centrale.csv"
    out=work.centrale
    dbms=csv replace;
    guessingrows=50;
    getnames=yes;
run;

proc import datafile="&data_path./hidroelectrica_macro_operationale.csv"
    out=work.macro
    dbms=csv replace;
    guessingrows=50;
    getnames=yes;
run;

proc import datafile="&data_path./hidroelectrica_segmente_2023_2025.csv"
    out=work.segmente
    dbms=csv replace;
    guessingrows=50;
    getnames=yes;
run;

proc import datafile="&data_path./hidroelectrica_cashflow_2024_2025.csv"
    out=work.cashflow
    dbms=csv replace;
    guessingrows=20;
    getnames=yes;
run;

title "FUNCTIA 1 — Verificare import: Date financiare consolidate 2021-2025";
proc print data=work.financiar noobs;
    var an venituri_totale profit_net ebitda marja_ebitda_pct roe_pct;
run;

title "FUNCTIA 1 — Verificare import: Centrale hidroelectrice (n=30)";
proc print data=work.centrale noobs;
    var nume rau judet putere_mw productie_gwh_an tip an_punere_functiune;
run;
title;

/* === FUNCTIA 2 — Formate definite de utilizator === */
proc format;
    /* An cu contextul hidrologic (indice precipitatii) */
    value an_hidro_fmt
        2021 = "2021 — Normal (98)"
        2022 = "2022 — Secetos (72)"
        2023 = "2023 — Ploios (118)"
        2024 = "2024 — Sub-normal (88)"
        2025 = "2025 — Secetos (71)";

    /* Marja EBITDA */
    value marja_fmt
        low  -<  50 = "Slaba  (<50%)"
        50   -<  65 = "Buna   (50-65%)"
        65   - high = "Exc.   (>65%)";

    /* Tip centrala (caracter) */
    value $ tip_fmt
        "acumulare"  = "Acumulare"
        "firul_apei" = "Firul apei"
        "fluvial"    = "Fluviala";

    /* Clasa de putere instalata */
    value putere_fmt
        low  -<  50 = "Mica   (<50 MW)"
        50   -< 200 = "Medie  (50-200 MW)"
        200  -< 500 = "Mare   (200-500 MW)"
        500  - high = "F. mare(>500 MW)";

    /* Profit net */
    value profit_fmt
        low              -< 3.5e9 = "Sub 3.5 mld RON"
        3.5e9            -< 5.0e9 = "3.5-5 mld RON"
        5.0e9            - high   = "Peste 5 mld RON";

    /* Varsta centrala */
    value varsta_fmt
        low -< 30 = "Moderna  (<=30 ani)"
        30  -< 50 = "Matura   (31-50 ani)"
        50  - high= "Veche    (>50 ani)";
run;

title "FUNCTIA 2 — Formate definite: Date financiare cu clasificare calitativa";
proc print data=work.financiar noobs label;
    var an marja_ebitda_pct profit_net;
    format an          an_hidro_fmt.
           marja_ebitda_pct marja_fmt.
           profit_net  profit_fmt.;
    label an              = "An (hidraulicitate)"
          marja_ebitda_pct= "Clasa marja EBITDA"
          profit_net      = "Clasa profit net";
run;
title;

proc sort data=work.financiar; by an; run;
proc sort data=work.macro;     by an; run;

data work.financiar_calc;
    merge work.financiar (in=inf)
          work.macro     (in=inm);
    by an;
    if inf and inm;

    /* Clasificare performanță operațională */
    if      marja_ebitda_pct >= 65 then performanta = "Excelenta";
    else if marja_ebitda_pct >= 50 then performanta = "Buna     ";
    else                                performanta = "Slaba    ";

    /* Scor compozit de sănătate financiară (0-100) */
    score_roe         = roe_pct          / 25  * 40;
    score_marja       = marja_ebitda_pct / 70  * 35;
    score_lichiditate = (rata_curenta    / 7.5) * 25;
    scor_compozit     = round(score_roe + score_marja + score_lichiditate, 0.01);

    /* Rata efectivă a impozitului */
    if profit_inainte_impozit > 0 then
        rata_impozit_ef = round((impozit_profit / profit_inainte_impozit) * 100, 0.01);
    else rata_impozit_ef = .;

    /* Cost mediu per angajat (mii RON/an) */
    cost_per_angajat_mii = round(cheltuieli_angajati / nr_angajati / 1000, 0.1);

    /* Clasificare an hidrologic */
    if      index_precipitatii >= 110 then tip_an_hidrologic = "Ploios  ";
    else if index_precipitatii >=  90 then tip_an_hidrologic = "Normal  ";
    else                                   tip_an_hidrologic = "Secetos ";

    /* CAGR venituri vs. 2021 */
    venituri_2021 = 6489297000;
    n_ani = an - 2021;
    if n_ani > 0 then
        cagr_venituri = round(((venituri_totale / venituri_2021) ** (1/n_ani) - 1) * 100, 0.01);
    else
        cagr_venituri = 0;

    /* Intensitate capital */
    intensitate_cap = round(imobilizari_corporale / venituri_totale, 0.0001);
run;

proc print data=work.financiar_calc noobs label;
    var an performanta tip_an_hidrologic scor_compozit
        rata_impozit_ef cost_per_angajat_mii cagr_venituri;
    format an an_hidro_fmt.;
run;

/* Subset 1: Ani cu profit net > 4 mld RON */
data work.ani_performanti;
    set work.financiar_calc;
    where profit_net > 4000000000;
run;

/* Subset 2: Ani cu marja EBITDA < 60% */
data work.ani_dificili;
    set work.financiar_calc;
    where marja_ebitda_pct < 60;
run;

/* Subset 3 & 4: Centrale mari și mici — OUTPUT simultan */
data work.centrale_mari work.centrale_mici;
    set work.centrale;
    if putere_mw >= 200 then output work.centrale_mari;
    else                     output work.centrale_mici;
run;

/* Subset 5: Centrale de acumulare din Vâlcea */
data work.centrale_valcea_acumulare;
    set work.centrale;
    where tip = "acumulare" and judet = "Valcea";
run;

title "FUNCTIA 4 — Subset: Ani performanti (profit net > 4 mld RON)";
proc print data=work.ani_performanti noobs;
    var an venituri_totale profit_net marja_ebitda_pct roe_pct performanta;
    format an an_hidro_fmt. marja_ebitda_pct marja_fmt.;
run;

title "FUNCTIA 4 — Subset: Ani dificili (marja EBITDA < 60%)";
proc print data=work.ani_dificili noobs;
    var an venituri_totale profit_net marja_ebitda_pct tip_an_hidrologic;
    format an an_hidro_fmt.;
run;

title "FUNCTIA 4 — Subset: Centrale mari (>=200 MW)";
proc print data=work.centrale_mari noobs;
    var nume rau judet putere_mw productie_gwh_an tip;
    format tip $tip_fmt. putere_mw putere_fmt.;
run;

title "FUNCTIA 4 — Subset: Centrale de acumulare din Valcea";
proc print data=work.centrale_valcea_acumulare noobs;
    var nume putere_mw productie_gwh_an an_punere_functiune;
run;
title;

data work.centrale_enriched;
    set work.centrale;

    /* Funcții numerice */
    factor_utilizare   = ROUND(productie_gwh_an / (putere_mw * 8760), 0.0001);
    productie_per_mw   = ROUND(productie_gwh_an / putere_mw, 0.01);
    putere_log         = ROUND(LOG(putere_mw), 0.0001);
    productie_sqrt     = ROUND(SQRT(productie_gwh_an), 0.1);
    dist_de_la_100mw   = ABS(putere_mw - 100);

    /* Vârsta centralei la 2025 */
    varsta_ani = 2025 - an_punere_functiune;

    /* Clasificare vârstă (SELECT-WHEN) */
    select;
        when (varsta_ani <= 30) grup_varsta = "Moderna  (<=30 ani)";
        when (varsta_ani <= 50) grup_varsta = "Matura   (31-50 ani)";
        otherwise               grup_varsta = "Veche    (>50 ani)";
    end;

    /* Funcții caracter */
    rau_upper    = UPCASE(rau);
    judet_upper  = UPCASE(judet);
    tip_abbrev   = SUBSTR(tip, 1, 3);
    lung_nume    = LENGTH(STRIP(nume));
    coordonate   = CATX(", ", PUT(ROUND(lat,0.0001), 8.4),
                              PUT(ROUND(lon,0.0001), 8.4));

    label factor_utilizare = "Factor de utilizare"
          productie_per_mw = "GWh produs per MW instalat"
          varsta_ani       = "Varsta la 2025 (ani)"
          grup_varsta      = "Categoria de varsta"
          coordonate       = "Coordonate geo (lat, lon)";
run;

title "FUNCTIA 5 — Functii SAS: Centrale imbogatite cu variabile derivate";
proc print data=work.centrale_enriched noobs label;
    var nume tip factor_utilizare productie_per_mw varsta_ani grup_varsta;
    format tip $tip_fmt. factor_utilizare 6.4;
run;

proc sort data=work.financiar_calc; by an; run;
proc sort data=work.macro;          by an; run;

data work.financiar_macro;
    merge work.financiar_calc (in=inf)
          work.macro          (in=inm);
    by an;
    if inf and inm;

    venituri_per_gwh     = ROUND(venituri_totale / (productie_hidro_gwh * 1000), 0.01);
    profit_per_gwh       = ROUND(profit_net      / (productie_hidro_gwh * 1000), 0.01);
    intensitate_muncii   = ROUND(nr_angajati / (productie_hidro_gwh / 1000), 0.1);
    pondere_hidro_in_nat = ROUND(productie_hidro_gwh / productie_nationala_gwh * 100, 0.01);

    label venituri_per_gwh     = "Venituri / GWh produs (RON/MWh)"
          profit_per_gwh       = "Profit net / GWh produs (RON/MWh)"
          intensitate_muncii   = "Angajati per TWh produs"
          pondere_hidro_in_nat = "Pondere Hidro in productia nationala (%)";
run;

title "FUNCTIA 6a — MERGE: Date financiare + Macro-operationale";
proc print data=work.financiar_macro noobs label;
    var an productie_hidro_gwh index_precipitatii
        venituri_per_gwh profit_per_gwh intensitate_muncii
        pondere_hidro_in_nat;
    format an an_hidro_fmt.;
run;

proc sql;
    title "FUNCTIA 6b — SQL JOIN: Segment x An x Pret energie";
    select s.an,
           s.segment,
           round(s.venituri_externe / 1e9, 0.01)
               as venituri_mld format=8.2 label="Venituri ext. (mld RON)",
           round(s.profit_inainte_impozit_segment / 1e9, 0.01)
               as profit_mld format=8.2 label="Profit segment (mld RON)",
           round(s.profit_inainte_impozit_segment /
                 s.venituri_externe * 100, 0.01)
               as marja_pct format=8.2 label="Marja (%)",
           m.pret_mediu_energie_ron_mwh label="Pret energie (RON/MWh)"
    from   work.segmente as s
    inner join work.macro as m on s.an = m.an
    where  s.tip = "consolidat"
    order  by s.an, s.segment;
    
proc sql;
    title "FUNCTIA 6b — SQL: Top 10 centrale dupa productie anuala";
    select c.nume,
           c.rau,
           c.judet,
           c.putere_mw             format=8.1 label="Putere (MW)",
           c.productie_gwh_an      format=comma8.0 label="Productie (GWh/an)",
           round(c.productie_gwh_an / c.putere_mw, 0.01)
                                   format=8.2 label="GWh/MW"
    from work.centrale as c
    order by c.productie_gwh_an desc;
quit;
title;

data work.financiar_indexat;
    set work.financiar_calc;

    /* Array indicatori de rentabilitate */
    array rent{4}     roe_pct roa_pct marja_neta_pct marja_ebitda_pct;
    array rent_idx{4} idx_roe idx_roa idx_marja_neta idx_marja_ebitda;
    array ref_rent{4} _temporary_ (16.24, 13.69, 48.02, 69.59);

    do i = 1 to 4;
        if ref_rent{i} ne 0 then
            rent_idx{i} = round(rent{i} / ref_rent{i} * 100, 0.01);
        else
            rent_idx{i} = .;
    end;

    /* Array cheltuieli: ponderi in venituri totale */
    array chelt{5}   cheltuieli_angajati cheltuieli_apa_uzinata
                     energie_achizitionata transport_distributie amortizare;
    array pondere{5} p_angajati p_apa_uzinata p_energie_ach p_transport p_amortizare;

    do j = 1 to 5;
        pondere{j} = round(chelt{j} / venituri_totale * 100, 0.01);
    end;

    /* Suma totala ponderi cheltuieli monitorizate */
    total_chelt_pct = sum(of pondere{*});

    drop i j;

    label idx_roe          = "Index ROE (2021=100)"
          idx_roa          = "Index ROA (2021=100)"
          idx_marja_neta   = "Index Marja neta (2021=100)"
          idx_marja_ebitda = "Index Marja EBITDA (2021=100)"
          p_angajati       = "Pondere cheltuieli angajati (%)"
          p_apa_uzinata    = "Pondere apa uzinata (%)"
          p_energie_ach    = "Pondere energie achizitionata (%)"
          p_transport      = "Pondere transport/distributie (%)"
          p_amortizare     = "Pondere amortizare (%)"
          total_chelt_pct  = "Total ponderi cheltuieli principale (%)";
run;

title "FUNCTIA 7 — Masive: Indici evolutie rentabilitate (baza 2021=100)";
proc print data=work.financiar_indexat noobs label;
    var an idx_roe idx_roa idx_marja_neta idx_marja_ebitda;
    format an an_hidro_fmt.;
run;

title "FUNCTIA 7 — Masive: Ponderi cheltuieli principale in venituri (%)";
proc print data=work.financiar_indexat noobs label;
    var an p_angajati p_apa_uzinata p_energie_ach p_transport p_amortizare total_chelt_pct;
    format an an_hidro_fmt.;
run;
title;

title "FUNCTIA 8a — PROC REPORT: Sinteza performantei financiare Hidroelectrica 2021-2025";
proc report data=work.financiar_macro nowd headline headskip;
    column an index_precipitatii productie_hidro_gwh
           venituri_totale profit_net marja_ebitda_pct roe_pct performanta;

    define an                   / group    "An"
                                  format=an_hidro_fmt. width=20;
    define index_precipitatii   / display  "Ind. Precip."     format=8.0       width=12;
    define productie_hidro_gwh  / display  "Prod. (GWh)"      format=comma8.0  width=12;
    define venituri_totale      / display  "Venituri (RON)"   format=comma22.0 width=24;
    define profit_net           / display  "Profit Net (RON)" format=comma22.0 width=24;
    define marja_ebitda_pct     / display  "Marja EBITDA"     format=8.2       width=12;
    define roe_pct              / display  "ROE (%)"          format=8.2       width=10;
    define performanta          / display  "Performanta"                       width=12;

    /* Linie de total cu medii */
    rbreak after / summarize dol dul;
    compute after;
        an = .;
        performanta = "MEDIE";
    endcomp;
run;

title "FUNCTIA 8b — PROC TABULATE: Statistici centrale pe tip si judet";
proc tabulate data=work.centrale_enriched;
    class  tip judet;
    var    putere_mw productie_gwh_an factor_utilizare varsta_ani;

    table  (tip all="TOTAL") * (judet all="Total"),
           (putere_mw productie_gwh_an factor_utilizare varsta_ani) *
           (N*f=4.0 Mean*f=8.1 Sum*f=comma10.0)
           / box="Tip Centrala / Judet" misstext="-";

    format tip $tip_fmt.;
    keylabel N="Nr." Mean="Medie" Sum="Total";
run;

title "FUNCTIA 8c — PROC PRINT: Cashflow Hidroelectrica 2024-2025";
proc print data=work.cashflow noobs label;
    var an tip profit_net numerar_din_exploatare
        investitii_corporale dividende_platite numerar_final;
    format profit_net numerar_din_exploatare investitii_corporale
           dividende_platite numerar_final comma22.0;
    label an                     = "An"
          tip                    = "Tip situatii"
          profit_net             = "Profit net (RON)"
          numerar_din_exploatare = "Cash din exploatare (RON)"
          investitii_corporale   = "Investitii corporale (RON)"
          dividende_platite      = "Dividende platite (RON)"
          numerar_final          = "Numerar final (RON)";
run;
title;

title "FUNCTIA 9a — PROC MEANS: Statistici descriptive indicatori financiari si operationali";
proc means data=work.financiar_macro
    n mean median std min max cv skewness kurtosis maxdec=2;
    var venituri_totale profit_net ebitda marja_ebitda_pct
        roe_pct roa_pct productie_hidro_gwh
        pret_mediu_energie_ron_mwh index_precipitatii;
    output out=work.stats_desc mean= median= std= cv= / autoname;
run;

title "FUNCTIA 9b — PROC FREQ: Distributia centralelor pe tip";
proc freq data=work.centrale_enriched;
    tables tip / nocum;
    format tip $tip_fmt.;
run;

title "FUNCTIA 9b — PROC FREQ: Tip centrala x Judet (tabel de contingenta)";
proc freq data=work.centrale_enriched;
    tables judet * tip / chisq nopercent norow;
    format tip $tip_fmt.;
run;

title "FUNCTIA 9b — PROC FREQ: Distributia centralelor pe grupa de varsta";
proc freq data=work.centrale_enriched;
    tables grup_varsta / nocum;
run;
title;

title "FUNCTIA 9c — PROC CORR: Matricea de corelatii indicatori financiari + operationali";
proc corr data=work.financiar_macro pearson spearman
    plots=matrix(histogram nvar=4);
    var profit_net venituri_totale productie_hidro_gwh
        index_precipitatii pret_mediu_energie_ron_mwh;
    ods output PearsonCorr  = work.corr_pearson
               SpearmanCorr = work.corr_spearman;
run;
title;

title "FUNCTIA 9d — PROC REG: Regresia profitului net in functie de factori macro";
proc reg data=work.financiar_macro
    plots(maxpoints=none)=(fitplot residuals);
    model profit_net = productie_hidro_gwh
                       pret_mediu_energie_ron_mwh
                       index_precipitatii /
                       vif tol r clb;
    output out=work.reg_macro_out
           predicted = profit_estimat
           residual  = rezidual
           student   = student_rezidual;
    title2 "Model: profit_net = b0 + b1*GWh + b2*Pret_energie + b3*Precip";
run;
title;

title "FUNCTIA 10a — SGPLOT: Evolutia financiara Hidroelectrica 2021-2025";
proc sgplot data=work.financiar_macro;
    vbarparm category=an response=venituri_totale /
             legendlabel="Venituri Totale"
             fillattrs=(color=CX1976D2) transparency=0.15;
    series x=an y=profit_net /
              legendlabel="Profit Net"
              lineattrs=(color=CX388E3C thickness=3 pattern=solid)
              markers markerattrs=(size=10 symbol=squarefilled color=CX388E3C);
    series x=an y=marja_ebitda_pct / y2axis
              legendlabel="Marja EBITDA (%)"
              lineattrs=(color=CXF57C00 thickness=3)
              markers markerattrs=(size=10 symbol=circlefilled color=CXF57C00);
    y2axis label="Marja EBITDA (%)" grid
           values=(40 50 60 70 80);
    yaxis  label="RON" grid;
    xaxis  label="An" values=(2021 to 2025 by 1);
    keylegend / location=inside position=topleft;
    format an 4.;
run;

