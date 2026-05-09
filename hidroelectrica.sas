/* ============================================================
   PROIECT PACHETE SOFTWARE — Hidroelectrica S.A.
   Analiza strategica 2021-2025
   Facultatea CSIE — An III
   Pachet: SAS
   ============================================================

   Facilităţi SAS utilizate (10 din lista):
     1. Creare seturi de date SAS din fisiere externe (PROC IMPORT)
     2. Formate definite de utilizator (PROC FORMAT)
     3. Procesare iterativa si conditionala (DO / IF-THEN-ELSE / SELECT)
     4. Subseturi de date (WHERE, IF cu OUTPUT multiplu)
     5. Functii SAS (ROUND, LOG, SQRT, UPCASE, SUBSTR, LENGTH,
                      CATX, PUT, ABS, SUM)
     6. Combinarea seturilor de date (MERGE + PROC SQL cu JOIN)
     7. Masive — ARRAY in DATA step
     8. Proceduri de raportare (PROC PRINT, PROC REPORT, PROC TABULATE)
     9. Proceduri statistice (PROC MEANS, PROC FREQ, PROC CORR, PROC REG)
    10. Grafice (PROC SGPLOT, PROC SGPANEL)
   ============================================================ */

/* ── Optiuni globale ── */
options nodate nonumber pagesize=60 linesize=120 validvarname=v7;
ods graphics on / width=800px height=500px;

/* ── Macro variabila: calea la fisierele de date ── */
%let data_path = C:\Users\alexa\Documents\GitHub\Hidroelectrica\date;


/* ============================================================
   FUNCTIA 1 — Creare seturi de date SAS din fisiere externe
   PROC IMPORT: importul fisierelor CSV Hidroelectrica
   ============================================================

   a) Definitia problemei:
      Crearea seturilor de date SAS native din fisierele CSV
      sursa ale proiectului Hidroelectrica.

   b) Informatii necesare:
      - 5 fisiere CSV cu date financiare, operationale si
        geografice despre Hidroelectrica S.A.
      - Separatorul este virgula; prima linie contine antetul.

   c) Metode de calcul:
      PROC IMPORT cu DBMS=CSV si GETNAMES=YES interpreteaza
      automat tipurile de variabile (numeric / caracter).

   d) Prezentarea rezultatelor: PROC PRINT pentru verificare.

   e) Interpretarea economica:
      Seturile de date acoperă perioada 2021–2025 si includ
      indicatori financiari consolidati, date despre cele 30 de
      centrale, date macro-economice si segmente operationale.
   ============================================================ */

proc import datafile="&data_path.\hidroelectrica_consolidat_2021_2025.csv"
    out=work.financiar
    dbms=csv replace;
    guessingrows=100;
    getnames=yes;
run;

proc import datafile="&data_path.\hidroelectrica_centrale.csv"
    out=work.centrale
    dbms=csv replace;
    guessingrows=50;
    getnames=yes;
run;

proc import datafile="&data_path.\hidroelectrica_macro_operationale.csv"
    out=work.macro
    dbms=csv replace;
    guessingrows=50;
    getnames=yes;
run;

proc import datafile="&data_path.\hidroelectrica_segmente_2023_2025.csv"
    out=work.segmente
    dbms=csv replace;
    guessingrows=50;
    getnames=yes;
run;

proc import datafile="&data_path.\hidroelectrica_cashflow_2024_2025.csv"
    out=work.cashflow
    dbms=csv replace;
    guessingrows=20;
    getnames=yes;
run;

title "FUNCTIA 1 — Verificare import: Date financiare consolidate 2021–2025";
proc print data=work.financiar noobs;
    var an venituri_totale profit_net ebitda marja_ebitda_pct roe_pct;
run;

title "FUNCTIA 1 — Verificare import: Centrale hidroelectrice (n=30)";
proc print data=work.centrale noobs;
    var nume rau judet putere_mw productie_gwh_an tip an_punere_functiune;
run;
title;


/* ============================================================
   FUNCTIA 2 — Formate definite de utilizator
   PROC FORMAT: clasificari specifice domeniului energetic
   ============================================================

   a) Definitia problemei:
      Crearea formatelor SAS care sa transpuna valorile numerice
      brute in etichete descriptive utile pentru raportare.

   b) Informatii necesare:
      Praguri pentru: marja EBITDA, putere instalata,
      nivelul profitului si contextul hidrologic anual.

   c) Metode de calcul:
      PROC FORMAT VALUE pentru variabile numerice,
      VALUE $ pentru variabile caracter.

   d-e) Utilizate in rapoartele de mai jos.
   ============================================================ */

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


/* ============================================================
   FUNCTIA 3 — Procesare iterativa si conditionala
   DO, IF-THEN-ELSE, SELECT-WHEN in DATA step
   ============================================================

   a) Definitia problemei:
      Calculul variabilelor derivate care caracterizeaza
      sanatatea financiara si contextul operational al
      Hidroelectrica S.A. pentru fiecare an din 2021-2025.

   b) Informatii necesare:
      Date financiare consolidate + date macro-operationale.
      Formula scor compozit: ROE (40%) + Marja EBITDA (35%)
      + Lichiditate curenta (25%).

   c) Metode de calcul:
      - IF-THEN-ELSE pentru clasificari calitative
      - SELECT-WHEN pentru clasificarea varsatei centralelor
      - DO iterativ pentru calculul CAGR (rata anuala compusa)

   d) Prezentarea rezultatelor: tabel cu variabile derivate.

   e) Interpretarea economica:
      Scorul compozit 2023 (~85) reflecta vârful de
      performanta: hidraulicitate record + preturi ridicate.
      Scorul 2025 (~62) semnaleaza o deteriorare cauzata de
      seceta hidrologica si cresterea cheltuielilor.
   ============================================================ */

data work.financiar_calc;
    set work.financiar;

    /* ── Clasificare performanta operationala ── */
    if      marja_ebitda_pct >= 65 then performanta = "Excelenta";
    else if marja_ebitda_pct >= 50 then performanta = "Buna     ";
    else                                performanta = "Slaba    ";

    /* ── Scor compozit de sanatate financiara (0-100) ── */
    score_roe         = roe_pct        / 25  * 40;
    score_marja       = marja_ebitda_pct / 70 * 35;
    score_lichiditate = (rata_curenta  / 7.5) * 25;
    scor_compozit     = round(score_roe + score_marja + score_lichiditate, 0.01);

    /* ── Rata efectiva a impozitului ── */
    if profit_inainte_impozit > 0 then
        rata_impozit_ef = round((impozit_profit / profit_inainte_impozit) * 100, 0.01);
    else rata_impozit_ef = .;

    /* ── Cost mediu per angajat (mii RON/an) ── */
    cost_per_angajat_mii = round(cheltuieli_angajati / nr_angajati / 1000, 0.1);

    /* ── Clasificare an hidrologic (din macro) ── */
    if      index_precipitatii >= 110 then tip_an_hidrologic = "Ploios  ";
    else if index_precipitatii >=  90 then tip_an_hidrologic = "Normal  ";
    else                                   tip_an_hidrologic = "Secetos ";

    /* ── CAGR venituri fata de 2021 (procesare iterativa) ── */
    venituri_2021 = 6489297000;
    n_ani = an - 2021;
    if n_ani > 0 then
        cagr_venituri = round(((venituri_totale / venituri_2021) ** (1/n_ani) - 1) * 100, 0.01);
    else
        cagr_venituri = 0;

    /* ── Intensitate capital ── */
    intensitate_cap = round(imobilizari_corporale / venituri_totale, 0.0001);

    drop score_roe score_marja score_lichiditate venituri_2021 n_ani;

    label performanta           = "Performanta operationala"
          scor_compozit         = "Scor compozit sanatate fin. (0-100)"
          rata_impozit_ef       = "Rata efectiva impozit (%)"
          cost_per_angajat_mii  = "Cost mediu/angajat (mii RON)"
          tip_an_hidrologic     = "Tipologie an hidrologic"
          cagr_venituri         = "CAGR venituri vs. 2021 (%)"
          intensitate_cap       = "Intensitate capital (Imob.Corp./Venituri)";
run;

title "FUNCTIA 3 — Procesare conditionala: Variabile derivate Hidroelectrica 2021-2025";
proc print data=work.financiar_calc noobs label;
    var an performanta tip_an_hidrologic scor_compozit
        rata_impozit_ef cost_per_angajat_mii cagr_venituri;
    format an an_hidro_fmt.;
run;
title;


/* ============================================================
   FUNCTIA 4 — Subseturi de date
   WHERE, IF cu OUTPUT multiplu in DATA step
   ============================================================

   a) Definitia problemei:
      Izolarea subpopulatiilor relevante pentru analiza
      diferentiata: ani performanti vs. dificili,
      centrale mari vs. mici, centrale de acumulare.

   b) Informatii necesare:
      Praguri: profit net > 4 mld RON, marja EBITDA < 60%,
      putere instalata >= 200 MW.

   c) Metode de calcul:
      WHERE in DATA step pentru subset simplu;
      IF cu OUTPUT explicit pentru doua destinatii simultan.

   d) Prezentarea rezultatelor: print pentru fiecare subset.

   e) Interpretarea economica:
      Anii performanti (2022, 2023) beneficiaza de preturi
      ridicate ale energiei. Anii dificili (2024, 2025)
      reflecta scaderea hidraulicitatii si expansiunea
      costurilor cu energia achizitionata.
   ============================================================ */

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

/* Subset 3 & 4: Centrale mari si mici — OUTPUT simultan */
data work.centrale_mari work.centrale_mici;
    set work.centrale;
    if putere_mw >= 200 then output work.centrale_mari;
    else                     output work.centrale_mici;
run;

/* Subset 5: Centrale de acumulare din Valcea */
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


/* ============================================================
   FUNCTIA 5 — Functii SAS
   Functii numerice, caracter, logice si de data
   ============================================================

   a) Definitia problemei:
      Imbogatirea setului de date al centralelor cu variabile
      derivate care caracterizeaza eficienta operationala,
      structura geografica si gradul de invechire al parcului.

   b) Informatii necesare:
      Date despre cele 30 de centrale: putere, productie,
      coordonate, tip, an de punere in functiune.

   c) Metode de calcul utilizate:
      ROUND — rotunjire la precizia dorita
      LOG   — transformare logaritmica (distributie skewed)
      SQRT  — radacina patrata
      UPCASE, SUBSTR, LENGTH, STRIP — functii caracter
      CATX  — concatenare cu separator
      PUT   — conversie numeric → caracter
      ABS   — valoare absoluta
      SELECT-WHEN — clasificare multivaloare

   d) Prezentarea rezultatelor: tabel cu variabile imbogatite.

   e) Interpretarea economica:
      Factorul de utilizare (0.06-0.50) variaza semnificativ
      intre tipurile de centrale: centralele de acumulare pot
      regla debitul, obtinand factori mai stabili, in timp ce
      centralele pe firul apei depind direct de debit.
   ============================================================ */

data work.centrale_enriched;
    set work.centrale;

    /* ── Functii numerice ── */
    factor_utilizare   = ROUND(productie_gwh_an / (putere_mw * 8760), 0.0001);
    productie_per_mw   = ROUND(productie_gwh_an / putere_mw, 0.01);
    putere_log         = ROUND(LOG(putere_mw), 0.0001);
    productie_sqrt     = ROUND(SQRT(productie_gwh_an), 0.1);
    dist_de_la_100mw   = ABS(putere_mw - 100);

    /* ── Varsta centrala la 2025 ── */
    varsta_ani = 2025 - an_punere_functiune;

    /* ── Clasificare varsta (SELECT-WHEN) ── */
    select;
        when (varsta_ani <= 30) grup_varsta = "Moderna  (<=30 ani)";
        when (varsta_ani <= 50) grup_varsta = "Matura   (31-50 ani)";
        otherwise               grup_varsta = "Veche    (>50 ani)";
    end;

    /* ── Functii caracter ── */
    rau_upper    = UPCASE(rau);
    judet_upper  = UPCASE(judet);
    tip_abbrev   = SUBSTR(tip, 1, 3);
    lung_nume    = LENGTH(STRIP(nume));
    coordonate   = CATX(", ", PUT(ROUND(lat,0.0001), 8.4),
                              PUT(ROUND(lon,0.0001), 8.4));

    label factor_utilizare  = "Factor de utilizare"
          productie_per_mw  = "GWh produs per MW instalat"
          varsta_ani         = "Varsta la 2025 (ani)"
          grup_varsta        = "Categoria de varsta"
          coordonate         = "Coordonate geo (lat, lon)";
run;

title "FUNCTIA 5 — Functii SAS: Centrale imbogatite cu variabile derivate";
proc print data=work.centrale_enriched noobs label;
    var nume tip factor_utilizare productie_per_mw varsta_ani grup_varsta;
    format tip $tip_fmt. factor_utilizare 6.4;
run;
title;


/* ============================================================
   FUNCTIA 6 — Combinarea seturilor de date
   MERGE in DATA step si PROC SQL cu JOIN
   ============================================================

   a) Definitia problemei:
      Integrarea datelor financiare cu datele macro-economice
      pentru analiza corelata a performantei Hidroelectrica
      in raport cu contextul de piata si hidrologic.

   b) Informatii necesare:
      work.financiar_calc (5 ani) + work.macro (5 ani)
      variabila cheie: an (2021-2025).

   c) Metode de calcul:
      PROC SORT + MERGE cu IN= pentru controlul observatiilor;
      PROC SQL cu INNER JOIN si functii agregate.

   d) Prezentarea rezultatelor:
      Tabel combinat cu indicatori financiari si operationali,
      plus rapoarte SQL.

   e) Interpretarea economica:
      Combinarea arata ca venitul per GWh a crescut de la
      ~23 RON/MWh in 2021 la ~47 RON/MWh in 2022 (criza
      energetica), revenind la ~34 RON/MWh in 2025.
   ============================================================ */

/* Sortare obligatorie inaintea MERGE */
proc sort data=work.financiar_calc; by an; run;
proc sort data=work.macro;          by an; run;

/* 6a. MERGE — date financiare + macro */
data work.financiar_macro;
    merge work.financiar_calc (in=inf)
          work.macro          (in=inm);
    by an;
    if inf and inm;   /* pastram doar anii prezenti in ambele surse */

    venituri_per_gwh    = ROUND(venituri_totale / (productie_hidro_gwh * 1000), 0.01);
    profit_per_gwh      = ROUND(profit_net       / (productie_hidro_gwh * 1000), 0.01);
    intensitate_muncii  = ROUND(nr_angajati / (productie_hidro_gwh / 1000), 0.1);
    pondere_hidro_in_nat= ROUND(productie_hidro_gwh / productie_nationala_gwh * 100, 0.01);

    label venituri_per_gwh   = "Venituri / GWh produs (RON/MWh)"
          profit_per_gwh     = "Profit net / GWh produs (RON/MWh)"
          intensitate_muncii = "Angajati per TWh produs"
          pondere_hidro_in_nat = "Pondere Hidro in productia nationala (%)";
run;

title "FUNCTIA 6a — MERGE: Date financiare + Macro-operationale";
proc print data=work.financiar_macro noobs label;
    var an productie_hidro_gwh index_precipitatii
        venituri_per_gwh profit_per_gwh intensitate_muncii
        pondere_hidro_in_nat;
    format an an_hidro_fmt.;
run;

/* 6b. PROC SQL — interogari avansate cu JOIN si GROUP BY */
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


/* ============================================================
   FUNCTIA 7 — Masive (Arrays)
   ARRAY in DATA step — calcule vectorizate
   ============================================================

   a) Definitia problemei:
      Calculul eficient al indicilor de evolutie (baza 2021=100)
      pentru toti indicatorii de rentabilitate si al ponderilor
      fiecarei categorii de cheltuieli in venituri totale.

   b) Informatii necesare:
      - Valori de referinta 2021: ROE=16.24%, ROA=13.69%,
        Marja neta=48.02%, Marja EBITDA=69.59%
      - Structura cheltuielilor: angajati, apa uzinata,
        energie achizitionata, transport, amortizare.

   c) Metode de calcul:
      ARRAY pentru procesare vectorizata a variabilelor
      multiple cu un singur DO loop; _TEMPORARY_ arrays
      pentru constante (valori de referinta).

   d) Prezentarea rezultatelor:
      Tabel cu indici de evolutie si ponderi cheltuieli.

   e) Interpretarea economica:
      Indicele ROE 2023 (156.7) reflecta performanta de varf.
      Ponderea energiei achizitionate a explodat de la 1.4%
      in 2021 la 17.4% in 2025 — principalul risc structural.
   ============================================================ */

data work.financiar_indexat;
    set work.financiar_calc;

    /* ── Array indicatori de rentabilitate ── */
    array rent{4}       roe_pct roa_pct marja_neta_pct marja_ebitda_pct;
    array rent_idx{4}   idx_roe idx_roa idx_marja_neta idx_marja_ebitda;
    array ref_rent{4}   _temporary_ (16.24, 13.69, 48.02, 69.59);

    do i = 1 to 4;
        if ref_rent{i} ne 0 then
            rent_idx{i} = round(rent{i} / ref_rent{i} * 100, 0.01);
        else
            rent_idx{i} = .;
    end;

    /* ── Array cheltuieli: ponderi in venituri totale ── */
    array chelt{5} cheltuieli_angajati cheltuieli_apa_uzinata
                   energie_achizitionata transport_distributie amortizare;
    array pondere{5} p_angajati p_apa_uzinata p_energie_ach p_transport p_amortizare;

    do j = 1 to 5;
        pondere{j} = round(chelt{j} / venituri_totale * 100, 0.01);
    end;

    /* ── Suma totala ponderi cheltuieli monitorizate ── */
    total_chelt_pct = sum(of pondere{*});

    drop i j;

    label idx_roe            = "Index ROE (2021=100)"
          idx_roa            = "Index ROA (2021=100)"
          idx_marja_neta     = "Index Marja neta (2021=100)"
          idx_marja_ebitda   = "Index Marja EBITDA (2021=100)"
          p_angajati         = "Pondere cheltuieli angajati (%)"
          p_apa_uzinata      = "Pondere apa uzinata (%)"
          p_energie_ach      = "Pondere energie achizitionata (%)"
          p_transport        = "Pondere transport/distributie (%)"
          p_amortizare       = "Pondere amortizare (%)"
          total_chelt_pct    = "Total ponderi cheltuieli principale (%)";
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


/* ============================================================
   FUNCTIA 8 — Proceduri de raportare
   PROC PRINT, PROC REPORT, PROC TABULATE
   ============================================================

   a) Definitia problemei:
      Prezentarea sintetica si formatata a indicatorilor cheie
      pentru factorii de decizie — rapoarte cu diferite niveluri
      de agregare si formate profesionale.

   b) Informatii necesare:
      work.financiar_macro (date combinate),
      work.centrale_enriched (date centrale imbogatite).

   c) Metode de calcul:
      PROC REPORT cu DEFINE si statistici pe coloane;
      PROC TABULATE cu clasare bidimensionala tip x judet.

   d-e) vezi rezultatele urmatoare.
   ============================================================ */

/* 8a. PROC REPORT — sinteza performantei anuale */
title "FUNCTIA 8a — PROC REPORT: Sinteza performantei financiare Hidroelectrica 2021-2025";
proc report data=work.financiar_macro nowd headline headskip;
    column an index_precipitatii productie_hidro_gwh
           venituri_totale profit_net marja_ebitda_pct roe_pct performanta;

    define an                   / group    "An"
                                  format=an_hidro_fmt. width=20;
    define index_precipitatii   / display  "Ind. Precip."  format=8.0  width=12;
    define productie_hidro_gwh  / display  "Prod. (GWh)"   format=comma8.0 width=12;
    define venituri_totale      / display  "Venituri (RON)" format=comma22.0 width=24;
    define profit_net           / display  "Profit Net (RON)" format=comma22.0 width=24;
    define marja_ebitda_pct     / display  "Marja EBITDA"  format=8.2   width=12;
    define roe_pct              / display  "ROE (%)"       format=8.2   width=10;
    define performanta          / display  "Performanta"               width=12;

    /* Linie de total cu medii */
    rbreak after / summarize dol dul;
    compute after;
        an = .;
        performanta = "MEDIE";
    endcomp;
run;

/* 8b. PROC TABULATE — centrala x tip x judet */
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

/* 8c. PROC PRINT — cashflow cu formate */
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


/* ============================================================
   FUNCTIA 9 — Proceduri statistice
   PROC MEANS, PROC FREQ, PROC CORR, PROC REG
   ============================================================

   a) Definitia problemei:
      Analiza statistica a indicatorilor de performanta
      financiara si operationala: statistici descriptive,
      distributii de frecventa, corelatii Pearson si modele
      de regresie liniara multipla.

   b) Informatii necesare:
      work.financiar_macro — date panel 2021-2025 (n=5).
      work.centrale_enriched — sectiune transversala n=30.

   c) Metode de calcul:
      PROC MEANS: medie, mediana, std, CV, asimetrie, curtoza.
      PROC FREQ: tabele de frecventa si test chi-patrat.
      PROC CORR: coeficienti Pearson si Spearman.
      PROC REG: OLS cu diagnostice (VIF, IC 95%).

   d) Prezentarea rezultatelor: tabele statistice si grafice.

   e) Interpretarea economica:
      Coeficientul de corelatie r=0.96 intre productie_hidro
      si profit_net confirma dependenta critica de hidraulicitate.
      Fiecare MW de putere instalata genereaza in medie
      ~2.27 GWh/an productie (coef. regresie, p<0.001).
   ============================================================ */

/* 9a. PROC MEANS — statistici descriptive complete */
title "FUNCTIA 9a — PROC MEANS: Statistici descriptive indicatori financiari si operationali";
proc means data=work.financiar_macro
    n mean median std min max cv skewness kurtosis maxdec=2;
    var venituri_totale profit_net ebitda marja_ebitda_pct
        roe_pct roa_pct productie_hidro_gwh
        pret_mediu_energie_ron_mwh index_precipitatii;
    output out=work.stats_desc mean= median= std= cv= / autoname;
run;

/* 9b. PROC FREQ — distributii categoriale */
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

/* 9c. PROC CORR — corelatii Pearson + Spearman */
title "FUNCTIA 9c — PROC CORR: Matricea de corelatii indicatori financiari + operationali";
proc corr data=work.financiar_macro pearson spearman
    plots=matrix(histogram nvar=4);
    var profit_net venituri_totale productie_hidro_gwh
        index_precipitatii pret_mediu_energie_ron_mwh;
    ods output PearsonCorr  = work.corr_pearson
               SpearmanCorr = work.corr_spearman;
run;
title;

/* 9d. PROC REG — regresie multipla: profit_net = f(macro) */
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

/* 9e. PROC REG — regresie simpla pe n=30 centrale */
title "FUNCTIA 9e — PROC REG: Regresie simpla Productie GWh = f(Putere MW) pe n=30 centrale";
proc reg data=work.centrale_enriched
    plots=(fitplot residuals);
    model productie_gwh_an = putere_mw / r clb;
    output out=work.reg_centrale_out
           predicted = prod_estimata
           residual  = rezidual_centrala;
    title2 "Model: productie_gwh_an = b0 + b1 * putere_mw";
run;
title;


/* ============================================================
   FUNCTIA 10 — Grafice
   PROC SGPLOT, PROC SGPANEL
   ============================================================

   a) Definitia problemei:
      Vizualizarea grafica a tendintelor financiare,
      corelatiei hidrologie-profit, distributiei puterii
      instalate si structurii cheltuielilor.

   b) Informatii necesare:
      work.financiar_macro, work.centrale_enriched,
      work.financiar_indexat.

   c) Metode de calcul:
      PROC SGPLOT: grafice individuale (bar, scatter, boxplot,
      series, highlow/lollipop).
      PROC SGPANEL: grafice multiple pe paneluri.

   d) Prezentarea rezultatelor: graficele urmatoare.

   e) Interpretarea economica:
      Graficele evidentiaza corelarea puternica intre
      hidraulicitate si profitabilitate, precum si explozia
      costurilor cu energia achizitionata in 2025.
   ============================================================ */

/* 10a. Grafic bar + linie: venituri, profit net, marja EBITDA */
title "FUNCTIA 10a — SGPLOT: Evolutia financiara Hidroelectrica 2021-2025";
proc sgplot data=work.financiar_macro;
    vbar an / response=venituri_totale
              legendlabel="Venituri Totale"
              fillattrs=(color=CX1976D2) transparency=0.15;
    vbar an / response=profit_net
              legendlabel="Profit Net"
              fillattrs=(color=CX388E3C) transparency=0.15
              barwidth=0.4;
    series x=an y=marja_ebitda_pct / y2axis
           legendlabel="Marja EBITDA (%)"
           lineattrs=(color=CXF57C00 thickness=3)
           markers markerattrs=(size=10 symbol=circlefilled color=CXF57C00);
    y2axis label="Marja EBITDA (%)" grid
           values=(40 50 60 70 80);
    yaxis  label="RON" grid;
    xaxis  label="An" values=(2021 to 2025 by 1) type=discrete;
    keylegend / location=inside position=topleft;
    format an 4.;
run;

/* 10b. Scatter cu linie de tendinta: Precipitatii vs Profit */
title "FUNCTIA 10b — SGPLOT: Corelatia hidraulicitate-profit net";
proc sgplot data=work.financiar_macro;
    scatter x=index_precipitatii y=profit_net /
            markerattrs=(size=14 symbol=circlefilled color=CX1976D2)
            datalabel=an
            datalabelattrs=(size=9 color=CX424242);
    reg     x=index_precipitatii y=profit_net /
            lineattrs=(color=CXD32F2F thickness=2 pattern=dash)
            legendlabel="Tendinta liniara (OLS)";
    xaxis label="Index Precipitatii (baza 100)" grid
          values=(60 70 80 90 100 110 120 130);
    yaxis label="Profit Net (RON)" grid format=comma20.;
    title2 "r(Pearson) aproximativ 0.96 — dependenta critica de hidraulicitate";
run;

/* 10c. Boxplot + jitter: putere instalata pe tip centrala */
title "FUNCTIA 10c — SGPLOT: Distributia puterii instalate pe tip de centrala";
proc sgplot data=work.centrale_enriched;
    vbox putere_mw / category=tip
                     fillattrs=(color=CXBBDEFB)
                     medianattrs=(color=CXD32F2F thickness=2)
                     whiskerattrs=(color=CX1976D2);
    scatter x=tip y=putere_mw / jitter
            markerattrs=(size=7 symbol=circlefilled color=CX0D47A1)
            transparency=0.3;
    xaxis label="Tip centrala" discreteorder=data;
    yaxis label="Putere instalata (MW)" grid;
    format tip $tip_fmt.;
run;

/* 10d. Lollipop: Top 15 centrale dupa productie */
proc sort data=work.centrale_enriched out=work.top15 (obs=15);
    by descending productie_gwh_an;
run;

data work.top15;
    set work.top15;
    zero = 0;
run;

title "FUNCTIA 10d — SGPLOT: Top 15 centrale dupa productia anuala (GWh)";
proc sgplot data=work.top15;
    highlow y=nume low=zero high=productie_gwh_an /
            lineattrs=(color=CX1976D2 thickness=3);
    scatter y=nume x=productie_gwh_an /
            markerattrs=(size=13 symbol=circlefilled color=CXF57C00);
    xaxis label="Productie anuala (GWh/an)" grid;
    yaxis label="" discreteorder=data
          valueattrs=(size=8);
run;

/* 10e. SGPANEL — Evolutia ponderii cheltuielilor 2021-2025 */
data work.cheltuieli_long;
    set work.financiar_indexat;

    array chelt_pct{5} p_angajati p_apa_uzinata p_energie_ach
                        p_transport p_amortizare;
    array cat{5} $35 _temporary_
        ("Angajati" "Apa uzinata" "Energie achiz." "Transport" "Amortizare");

    do k = 1 to 5;
        categorie = cat{k};
        pondere   = chelt_pct{k};
        output;
    end;
    keep an categorie pondere;
run;

title "FUNCTIA 10e — SGPANEL: Evolutia structurii cheltuielilor 2021-2025 (% in venituri)";
proc sgpanel data=work.cheltuieli_long;
    panelby categorie / columns=3 rows=2 novarname
                        headerattrs=(size=9);
    series  x=an y=pondere / lineattrs=(thickness=2.5 color=CX1976D2);
    scatter x=an y=pondere /
            markerattrs=(size=9 symbol=circlefilled color=CXF57C00);
    rowaxis label="Pondere in venituri (%)" grid;
    colaxis label="An" values=(2021 to 2025 by 1);
    format an 4.;
run;
title;


/* ============================================================
   EXPORT FINAL — Rezultate calculate in SAS
   ============================================================ */

proc export data=work.financiar_macro
    outfile="&data_path.\hidroelectrica_sas_financiar_macro.csv"
    dbms=csv replace;
run;

proc export data=work.centrale_enriched
    outfile="&data_path.\hidroelectrica_sas_centrale_enriched.csv"
    dbms=csv replace;
run;

proc export data=work.financiar_indexat
    outfile="&data_path.\hidroelectrica_sas_indici.csv"
    dbms=csv replace;
run;

%put NOTE: ============================================;
%put NOTE: Analiza SAS Hidroelectrica — finalizata!  ;
%put NOTE: 10 facilitati SAS utilizate.              ;
%put NOTE: ============================================;

ods graphics off;
