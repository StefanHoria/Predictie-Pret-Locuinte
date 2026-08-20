# Predicția prețului unei locuințe folosind inteligența artificială

### Studiu comparativ între regresia liniară, rețelele neuronale și metodele bazate pe arbori, pe date colectate din piața imobiliară din România

**Lucrare de practică — Anul III**

---

## Cuprins

1. [Introducere](#1-introducere)
2. [Fundamente teoretice](#2-fundamente-teoretice)
3. [Colectarea datelor](#3-colectarea-datelor)
4. [Analiza exploratorie a datelor](#4-analiza-exploratorie-a-datelor)
5. [Preprocesare și inginerie de caracteristici](#5-preprocesare-și-inginerie-de-caracteristici)
6. [Modelele implementate](#6-modelele-implementate)
7. [Rezultate și comparație](#7-rezultate-și-comparație)
8. [Probleme întâmpinate și diagnosticarea lor](#8-probleme-întâmpinate-și-diagnosticarea-lor)
9. [Concluzii](#9-concluzii)
10. [Bibliografie](#10-bibliografie)
11. [Anexe](#11-anexe)

---

## 1. Introducere

### 1.1. Contextul problemei

Evaluarea unei locuințe este o problemă economică cu miză practică ridicată. Cumpărătorii vor să știe dacă un preț cerut este rezonabil, vânzătorii vor să își poziționeze corect oferta, iar instituțiile financiare au nevoie de estimări pentru garanții ipotecare. În mod tradițional, evaluarea se face de către experți umani, pe baza experienței și a comparației cu tranzacții similare — un proces subiectiv, lent și greu de scalat.

Problema se pretează natural la învățarea automată supervizată: există un număr mare de exemple observabile (anunțuri de vânzare), fiecare cu un set de caracteristici măsurabile (suprafață, număr de camere, localizare) și cu o valoare-țintă cunoscută (prețul cerut). Sarcina modelului este să învețe funcția care leagă caracteristicile de preț.

### 1.2. Obiectivele lucrării

Lucrarea urmărește patru obiective:

1. **Construirea unui set de date propriu**, colectat din piața imobiliară reală din România, în locul folosirii unui set public deja curățat.
2. **Implementarea și antrenarea unui model de regresie liniară** ca referință interpretabilă.
3. **Implementarea unei rețele neuronale în PyTorch**, cu bucla de antrenare scrisă explicit, și compararea ei cu modelul liniar.
4. **Evaluarea comparativă riguroasă**, care să explice nu doar *care* model este mai bun, ci și *de ce* și *unde* diferă comportamentul lor.

Un al cincilea obiectiv, apărut pe parcurs, a fost includerea unui al treilea model — Random Forest — pentru a verifica afirmația frecventă din literatură conform căreia metodele bazate pe arbori de decizie depășesc rețelele neuronale pe date tabulare de dimensiuni moderate.

### 1.3. Contribuții proprii

- Un **colector automat de date** (*scraper*) care extrage anunțuri prin interfața de programare publică a platformei OLX, cu partiționare a interogărilor pentru a depăși limita de paginare impusă de server.
- Un **set de date de 28.637 de anunțuri curățate**, acoperind 632 de localități din toate cele 41 de județe.
- O **analiză comparativă a trei familii de modele**, incluzând o metodă de analiză de sensibilitate care evidențiază diferențele calitative dintre ele, nu doar cele numerice.
- **Documentarea a trei erori metodologice** identificate și corectate pe parcursul lucrării, cu explicarea cauzelor.

### 1.4. Structura lucrării

Capitolul 2 prezintă noțiunile teoretice necesare. Capitolul 3 descrie colectarea datelor. Capitolul 4 conține analiza exploratorie. Capitolul 5 detaliază preprocesarea. Capitolul 6 descrie modelele. Capitolul 7 prezintă rezultatele comparative. Capitolul 8 este dedicat problemelor întâmpinate — capitol pe care îl considerăm cel mai instructiv din lucrare. Capitolul 9 sintetizează concluziile.

---

## 2. Fundamente teoretice

### 2.1. Învățarea automată supervizată

**Învățarea automată** (*machine learning*) este ramura inteligenței artificiale care studiază algoritmi capabili să își îmbunătățească performanța pe o sarcină prin expunerea la date, fără a fi programați explicit pentru fiecare caz particular.

În **învățarea supervizată**, algoritmul primește un set de perechi (intrare, ieșire dorită) și trebuie să învețe funcția care le leagă. Formal, dispunem de un set de antrenare

$$D = \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}$$

unde $x_i \in \mathbb{R}^p$ este vectorul de caracteristici al exemplului $i$, iar $y_i$ este valoarea-țintă. Scopul este găsirea unei funcții $f$ astfel încât $f(x) \approx y$ pentru exemple **noi**, nevăzute în timpul antrenării.

Când $y$ este o valoare continuă (cazul nostru: prețul în euro), problema se numește **regresie**. Când $y$ aparține unui set finit de categorii, problema se numește **clasificare**.

### 2.2. Caracteristici, țintă și generalizare

**Caracteristica** (*feature*) este o variabilă de intrare măsurabilă. În lucrarea de față: suprafața utilă, numărul de camere, etajul, perioada de construcție, localizarea.

**Ținta** (*target*) este variabila pe care o prezicem — prețul cerut.

**Generalizarea** este capacitatea modelului de a face predicții corecte pe date pe care nu le-a văzut la antrenare. Este singura măsură relevantă a calității unui model. Un model care reproduce perfect datele de antrenare, dar eșuează pe date noi, este inutil.

Pentru a măsura generalizarea, setul de date se împarte în:
- **set de antrenare** (*training set*) — pe care modelul își ajustează parametrii;
- **set de validare** — folosit pentru decizii în timpul antrenării (de exemplu, când să oprim antrenarea);
- **set de testare** (*test set*) — folosit **o singură dată**, la final, pentru a raporta performanța.

### 2.3. Supraînvățarea și subînvățarea

**Supraînvățarea** (*overfitting*) apare când modelul memorează particularitățile setului de antrenare, inclusiv zgomotul, în loc să învețe tiparele generale. Semnul caracteristic: performanță mult mai bună pe antrenare decât pe testare.

**Subînvățarea** (*underfitting*) apare când modelul este prea simplu pentru a capta structura reală a datelor. Semnul caracteristic: performanță slabă și pe antrenare, și pe testare.

Echilibrul dintre cele două este cunoscut drept **compromisul bias–varianță**. Un model cu bias ridicat (prea simplu) subînvață; un model cu varianță ridicată (prea flexibil) supraînvață.

**Regularizarea** este orice tehnică ce restrânge complexitatea modelului pentru a reduce supraînvățarea. Cea mai comună formă este penalizarea **L2** (*ridge*), care adaugă la funcția de cost un termen proporțional cu suma pătratelor parametrilor:

$$J_{\text{regularizat}} = J + \alpha \sum_j w_j^2$$

Parametrul $\alpha$ controlează intensitatea penalizării.

### 2.4. Regresia liniară

Regresia liniară presupune că ținta este o combinație liniară a caracteristicilor:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_p x_p$$

unde $w_0$ este termenul liber (*intercept*), iar $w_1, \ldots, w_p$ sunt **coeficienții** modelului.

Parametrii se determină prin **metoda celor mai mici pătrate**, minimizând suma pătratelor erorilor:

$$J(w) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Problema are soluție analitică:

$$\hat{w} = (X^T X)^{-1} X^T y$$

**Avantaje:** interpretabilitate directă (fiecare coeficient arată efectul unei unități de variabilă asupra țintei), antrenare instantanee, comportament previzibil.

**Limitări:** modelul poate reprezenta **doar relații liniare**. Dacă efectul real al unei variabile asupra țintei este neliniar — de exemplu în formă de U — regresia liniară nu îl poate capta, oricâte date i-am da. Această limitare structurală este centrală pentru lucrarea de față.

### 2.5. Multicoliniaritatea

**Multicoliniaritatea** apare când două sau mai multe caracteristici sunt puternic corelate între ele. În acest caz, modelul nu poate distinge contribuția fiecăreia: mai multe combinații de coeficienți produc predicții aproape identice.

Consecințele sunt: coeficienți instabili (se schimbă mult la modificări mici ale datelor), coeficienți cu semn contraintuitiv și imposibilitatea interpretării izolate a unui coeficient. Puterea predictivă a modelului nu este însă afectată semnificativ.

În setul nostru, suprafața și numărul de camere au corelație 0,75 — un exemplu clasic.

### 2.6. Rețelele neuronale artificiale

O **rețea neuronală de tip perceptron multistrat** (*Multi-Layer Perceptron*, MLP) este o funcție compusă din straturi succesive de transformări.

Fiecare **strat** aplică o transformare afină urmată de o funcție neliniară de activare:

$$h^{(l)} = \sigma\left(W^{(l)} h^{(l-1)} + b^{(l)}\right)$$

unde $W^{(l)}$ este matricea de ponderi a stratului $l$, $b^{(l)}$ vectorul de deplasare (*bias*), iar $\sigma$ funcția de activare.

**Funcția de activare** este elementul care conferă rețelei capacitatea de a reprezenta relații neliniare. Fără ea, compunerea mai multor straturi liniare ar rămâne o funcție liniară. Am folosit **ReLU** (*Rectified Linear Unit*):

$$\text{ReLU}(z) = \max(0, z)$$

Alegerea ei este motivată de simplitatea calculului derivatei și de faptul că nu suferă de fenomenul de *dispariție a gradientului* la valori mari, spre deosebire de funcțiile sigmoidă sau tangentă hiperbolică.

**Teorema aproximării universale** (Cybenko, 1989; Hornik, 1991) garantează că o rețea cu un singur strat ascuns și un număr suficient de neuroni poate aproxima orice funcție continuă pe un domeniu mărginit, cu precizie arbitrară. Teorema afirmă existența unei astfel de rețele, nu și că ea poate fi găsită prin antrenare cu date limitate — o distincție importantă în practică.

### 2.7. Antrenarea prin propagarea înapoi a erorii

**Funcția de cost** (*loss function*) măsoară cât de departe sunt predicțiile de valorile reale. Pentru regresie am folosit **eroarea pătratică medie**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Coborârea pe gradient** (*gradient descent*) este algoritmul de optimizare: parametrii se ajustează în direcția opusă gradientului funcției de cost:

$$w \leftarrow w - \eta \frac{\partial J}{\partial w}$$

unde $\eta$ este **rata de învățare** (*learning rate*).

**Propagarea înapoi a erorii** (*backpropagation*) este algoritmul care calculează eficient gradientul costului față de fiecare parametru al rețelei, aplicând regula lanțului de derivare de la stratul de ieșire către cel de intrare. Fără acest algoritm, antrenarea rețelelor adânci ar fi computațional imposibilă.

**Epoca** (*epoch*) este o parcurgere completă a setului de antrenare. **Lotul** (*batch*) este submulțimea de exemple procesată la un pas de actualizare a parametrilor. Am folosit loturi de 256 de exemple.

**Adam** (*Adaptive Moment Estimation*, Kingma & Ba, 2015) este optimizatorul folosit. Spre deosebire de coborârea pe gradient clasică, Adam menține medii mobile ale gradienților și ale pătratelor lor, adaptând rata de învățare separat pentru fiecare parametru. Aceasta accelerează convergența și reduce sensibilitatea la alegerea ratei inițiale.

**Oprirea timpurie** (*early stopping*) este o formă de regularizare: antrenarea se întrerupe când eroarea pe setul de validare încetează să se îmbunătățească timp de un număr prestabilit de epoci. Previne memorarea setului de antrenare.

### 2.8. Metode bazate pe arbori. Random Forest

Un **arbore de decizie** pentru regresie partiționează recursiv spațiul caracteristicilor prin praguri de tipul „suprafață < 55 m²", predicând pentru fiecare regiune finală (frunză) media valorilor-țintă din acea regiune. Un arbore adânc poate reprezenta relații foarte neregulate, dar supraînvață puternic.

**Random Forest** (Breiman, 2001) reduce supraînvățarea prin combinarea a numeroși arbori antrenați pe eșantioane aleatoare cu revenire din setul de date (*bootstrap aggregating* sau *bagging*), fiecare arbore considerând la fiecare divizare doar un subset aleator de caracteristici. Predicția finală este media predicțiilor individuale.

Metoda este atractivă pentru date tabulare deoarece nu necesită scalarea caracteristicilor, tratează natural interacțiunile dintre variabile și este robustă la valori extreme.

### 2.9. Metrici de evaluare

Fie $y_i$ valorile reale și $\hat{y}_i$ predicțiile, pentru $n$ exemple de test.

**Eroarea absolută medie (MAE):**
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
Se exprimă în unitatea țintei (euro) și este direct interpretabilă. Este robustă la valori extreme.

**Eroarea absolută mediană (MedAE):** mediana valorilor $|y_i - \hat{y}_i|$. Și mai robustă decât MAE; arată eroarea tipică, neinfluențată de câteva cazuri aberante.

**Rădăcina erorii pătratice medii (RMSE):**
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
Penalizează disproporționat erorile mari. Diferența mare dintre RMSE și MAE indică existența unor erori extreme.

**Eroarea procentuală absolută medie (MAPE):**
$$\text{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
Cea mai relevantă metrică pentru imobiliare: o eroare de 10.000 € înseamnă altceva pentru un apartament de 30.000 € decât pentru unul de 500.000 €.

**Coeficientul de determinare (R²):**
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
Proporția din varianța țintei explicată de model. Valoarea 1 înseamnă predicție perfectă, 0 înseamnă performanță egală cu predicția mediei, iar valorile negative înseamnă performanță **mai slabă** decât predicția mediei.

### 2.10. Scurgerea de informație

**Scurgerea de informație** (*data leakage*) apare când modelul are acces, în timpul antrenării, la informații care nu vor fi disponibile la momentul predicției reale — cel mai adesea, informații derivate din setul de testare.

Efectul este o performanță aparent excelentă la evaluare, urmată de eșec în utilizare reală. Este considerată cea mai frecventă eroare metodologică gravă din proiectele de învățare automată.

Prevenirea presupune ca **orice transformare care folosește ținta să fie calculată exclusiv pe setul de antrenare** și apoi aplicată setului de testare, nu învățată din acesta.

### 2.11. Codificarea variabilelor categoriale

Modelele numerice nu pot procesa direct valori textuale precum „Cluj-Napoca". Sunt necesare transformări.

**Codificarea one-hot** creează câte o coloană binară pentru fiecare valoare distinctă. Este simplă și nu introduce ordine artificială, dar devine impracticabilă la cardinalitate mare: 632 de localități ar produce 632 de coloane, majoritatea aproape goale.

**Codificarea prin țintă** (*target encoding*) înlocuiește categoria cu o statistică a țintei calculată pentru acea categorie — în cazul nostru, prețul mediu pe metru pătrat al localității. Produce o singură coloană numerică, dar prezintă două riscuri: scurgere de informație (dacă se calculează pe tot setul) și instabilitate pentru categoriile rare.

**Netezirea bayesiană** rezolvă al doilea risc, combinând media categoriei cu o valoare de rezervă, ponderat după numărul de observații:

$$\text{cod}(c) = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}_{\text{rezervă}}}{n_c + m}$$

unde $n_c$ este numărul de observații din categoria $c$, iar $m$ este un parametru de netezire care se interpretează ca număr de observații „virtuale" atribuite valorii de rezervă.

---

## 3. Colectarea datelor

### 3.1. Motivația unui set de date propriu

Seturile publice consacrate pentru predicția prețurilor imobiliare (*Ames Housing*, *California Housing*) descriu piețe din Statele Unite, cu caracteristici arhitecturale și tipare de preț care nu se transferă la piața românească. În plus, ele sunt deja curățate, ceea ce elimină tocmai partea de muncă în care apar deciziile metodologice interesante.

Am optat pentru colectarea directă a datelor de pe **OLX.ro**, cea mai mare platformă de anunțuri imobiliare din România.

### 3.2. Alegerea sursei și a metodei de acces

Am evaluat trei platforme:

| Platformă | Accesibilitate | Observații |
|---|---|---|
| Storia.ro | Redusă | Protecție anti-bot (Cloudflare / DataDome) |
| Imobiliare.ro | Medie | Necesită analiza structurii HTML |
| **OLX.ro** | **Ridicată** | **Expune o interfață JSON publică** |

Interfața web a OLX consumă un punct de acces care returnează date structurate:

```
https://www.olx.ro/api/v1/offers/?category_id=907&limit=40&offset=0
```

Avantajul față de analiza codului HTML este substanțial: datele vin deja structurate, cu tipuri corecte, și nu depind de modificări ale interfeței grafice.

Identificatorul de categorie a fost determinat prin căutarea parametrului `category_id` în codul sursă al paginii de listare. Categoriile relevante:

| Categorie | Identificator | Anunțuri disponibile |
|---|---:|---:|
| 1 cameră | 1163 | 10.659 |
| 2 camere | 1165 | 38.930 |
| 3 camere | 1167 | 26.083 |
| 4+ camere | 1169 | 6.152 |

### 3.3. Constrângerea de paginare și soluția prin partiționare

Serverul raportează 81.815 anunțuri disponibile, dar limitează parcurgerea la **1.000 de rezultate per interogare**, indiferent de valoarea parametrului de decalaj (*offset*). Cereri cu decalaj peste 1.040 returnează eroarea HTTP 400.

Soluția adoptată este **partiționarea spațiului de căutare**: în loc de o interogare unică, am executat 164 de interogări independente, corespunzând produsului cartezian dintre 41 de județe și 4 categorii de camere. Fiecare partiție se încadrează sub limită sau este plafonată individual, ceea ce permite acoperirea unei fracțiuni mult mai mari din total.

Partiționarea rezolvă simultan o a doua problemă. Numărul de camere **nu este prezent** în răspunsul serverului — este codificat în subcategorie. Deoarece fiecare interogare vizează o subcategorie cunoscută, numărul de camere al fiecărui anunț rezultă din partiția care l-a returnat.

Identificatorii de județ au fost determinați prin scanarea sistematică a valorilor 1–50 ale parametrului `region_id`. Rezultatul: 41 de valori valide, cu discontinuități (valorile 5, 14, 25, 44, 45 nu sunt utilizate).

### 3.4. Arhitectura colectorului

Programul este organizat pe trei niveluri de responsabilitate:

| Modul | Responsabilitate |
|---|---|
| `fetch.py` | Comunicare HTTP: construirea cererii, paginare, reîncercări |
| `storage.py` | Persistență: salvare și încărcare din format JSON |
| `run_scraper.py` | Orchestrare: ce se colectează și cu ce parametri |

Separarea permite modificarea formatului de stocare fără a atinge codul de rețea și invers.

**Paginarea** se oprește pe baza semnalului transmis de server: dispariția cheii `next` din secțiunea de legături a răspunsului. Această condiție funcționează corect și pentru partițiile mici, care se epuizează înainte de limita de 1.000. O limită proprie de siguranță previne buclele infinite în cazul unui comportament neașteptat al serverului.

**Deduplicarea** se realizează pe două niveluri, folosind identificatorul unic al anunțului: în interiorul fiecărei partiții și, ulterior, la nivelul întregii colecții. Necesitatea ei este demonstrabilă empiric — serverul inserează anunțuri promovate peste limita solicitată, iar acestea se repetă între pagini. Într-un test cu 3 pagini au fost primite 156 de anunțuri, dintre care 154 unice.

**Politica de reîncercare** distinge între erori temporare și permanente. Codurile 5xx (eroare de server) și 429 (prea multe cereri) declanșează reîncercare cu pauză crescătoare exponențial (2, 4, 8 secunde). Celelalte coduri 4xx indică o cerere greșită, care nu se va corecta prin repetare, și propagă imediat eroarea.

### 3.5. Considerații etice și tehnice

Colectarea respectă următoarele principii:

- **Identificare onestă**: antetul `User-Agent` declară explicit natura și scopul programului (`practica-student-scraper/0.1 (proiect educational)`), fără a imita un navigator web.
- **Limitarea ratei**: pauză de 1,5 secunde între cereri, rezultând într-o medie de aproximativ 1,3 cereri pe secundă.
- **Utilizare exclusiv educațională**: datele servesc doar antrenării modelelor din prezenta lucrare.
- **Neredistribuire**: setul de date brute nu este publicat și este exclus din controlul de versiuni.

### 3.6. Rezultatul colectării

| Indicator | Valoare |
|---|---|
| Data colectării | 20 august 2026 |
| Interogări executate | 164 (41 județe × 4 categorii) |
| Durata totală | 22,5 minute |
| Anunțuri colectate | **30.128** |
| Duplicate eliminate | 0 după deduplicare finală |
| Volum date brute | 212,6 MB |

---

## 4. Analiza exploratorie a datelor

### 4.1. Structura și completitudinea

Analiza gradului de completare a câmpurilor a determinat care variabile pot fi folosite:

| Câmp | Acoperire | Decizie |
|---|---:|---|
| Preț | 100,0% | Țintă |
| Suprafață utilă | 98,6% | Caracteristică principală |
| Etaj | 95,4% | Inclus, cu imputare |
| An construcție | 74,9% | Inclus, cu imputare și indicator |
| Compartimentare | **15,0%** | **Eliminat** |
| Coordonate geografice | 100,0% | Inclus, derivat |

Compartimentarea a fost eliminată: cu 85% valori lipsă, imputarea ar introduce mai mult zgomot decât informație.

Două observații privind natura datelor:

**Anul construcției nu este un an**, ci un interval: `înainte de 1977`, `1977–1990`, `1990–2000`, `după 2000`. Nu se poate calcula vechimea în ani; variabila a fost tratată ca ordinală, cu valorile 0–3 în ordine cronologică.

**Etajul include valori nenumerice**: `Parter`, `Demisol`, `Mansardă`, iar `10` semnifică de fapt „10 și peste". Am codificat demisolul cu −1 și parterul cu 0, poziții cu sens în ordinea verticală. Mansarda a fost lăsată nedefinită în coloana numerică, deoarece poziția ei relativă diferă de la clădire la clădire; forțarea unei valori ar fi însemnat inventarea unei informații inexistente.

**Coordonatele geografice** s-au dovedit precise la nivel de cartier, nu doar de localitate: pentru 1.445 de anunțuri din București există 1.086 de poziții distincte, cu rază de eroare declarată de 0–1 km. Aceasta face posibilă construirea unei variabile de centralitate.

### 4.2. Distribuția prețului

Distribuția prețului brut este puternic asimetrică la dreapta, cu coeficient de asimetrie **3,30**. Media (99.050 €) depășește considerabil mediana (85.000 €), iar valorile extreme ajung la 1.200.000 €.

După transformarea logaritmică, asimetria scade la **0,16**, distribuția devenind practic simetrică.

> **Figura 1** — `reports/figures/01_distributie_pret.png`

Constatarea are consecință directă asupra modelării. Regresia liniară presupune erori distribuite normal; pe prețul brut presupunerea este falsă, iar modelul va fi sistematic imprecis pe segmentul superior. Am adoptat, în consecință, **logaritmul prețului ca variabilă-țintă**.

### 4.3. Relația dintre suprafață și preț

Suprafața este cea mai puternic corelată variabilă cu prețul (r = 0,745). Reprezentarea grafică în coordonate liniare arată o relație crescătoare, dar cu dispersie care se amplifică odată cu suprafața. În coordonate dublu-logaritmice, relația devine aproximativ liniară, cu dispersie constantă.

> **Figura 2** — `reports/figures/02_pret_vs_suprafata.png`

Această observație a stat la baza deciziei de a logaritma și variabila de intrare, nu doar ținta (detaliat în secțiunea 8.1).

### 4.4. Efectul localizării

Localizarea este al doilea factor ca importanță. Prețul median pe metru pătrat variază cu un factor de **peste 3** între extreme:

| Localitate | €/m² median | Anunțuri |
|---|---:|---:|
| Cluj-Napoca | 3.000 | 888 |
| Brașov | 2.114 | 1.032 |
| Craiova | 2.048 | 1.088 |
| Florești | 1.980 | 424 |
| ... | | |
| Roman | 1.000 | 331 |
| Reșița | 949 | 140 |
| Hunedoara | 929 | 129 |

> **Figura 3** — `reports/figures/03_pret_per_mp_judete.png`

Două constatări merită subliniate.

**Florești**, comună limitrofă Clujului, are prețuri superioare celor din Timișoara — municipiu reședință de județ. Prețul nu reflectă caracteristicile localității în sine, ci **proximitatea față de un centru economic**.

**Județul București-Ilfov** ocupă abia poziția a opta după mediană și prezintă cea mai mare dispersie internă din întregul set. Explicația: județul agregă municipiul București cu comunele din Ilfov, cu prețuri mult mai scăzute. Constatarea demonstrează că nivelul județean este prea grosier pentru modelare.

### 4.5. Matricea de corelații

> **Figura 4** — `reports/figures/04_corelatii.png`

| Variabilă | Corelație cu prețul |
|---|---:|
| Suprafață | +0,745 |
| Număr camere | +0,554 |
| An construcție (ordinal) | +0,273 |
| Etaj | **+0,076** |
| Latitudine | −0,055 |
| Longitudine | −0,063 |

Corelația **suprafață ↔ camere = 0,75** indică multicoliniaritate, cu implicațiile discutate în secțiunea 2.5.

Coordonatele geografice brute au corelație practic nulă cu prețul. Rezultatul este așteptat: nu există o tendință liniară pe hartă, deoarece atât Clujul (nord-vest) cât și Bucureștiul (sud) sunt scumpe, iar Vasluiul (est) este ieftin. Informația geografică există, dar nu în formă liniar exploatabilă — de aici necesitatea unei transformări (secțiunea 5.2).

### 4.6. Etajul: o relație neliniară

Corelația liniară dintre etaj și preț este 0,076 — practic nulă. Concluzia superficială ar fi eliminarea variabilei.

Analiza pe categorii arată însă altceva:

> **Figura 5** — `reports/figures/05_factori.png`

Prețul pe metru pătrat atinge **minimul la etajul 4** (aproximativ 1.380 €/m²) și **maximul la etajele 6–10** (aproximativ 1.800 €/m²), o diferență de circa 25%.

Explicația ține de particularitățile fondului locativ românesc. Blocurile construite în perioada 1960–1990 au predominant regim de înălțime P+4, ceea ce face ca etajul 4 să fie ultimul — cu problemele asociate terasei și, frecvent, fără lift. Un apartament situat peste etajul 5 se află în mod necesar într-o clădire mai nouă, dotată cu lift.

Relația are, așadar, formă de U, iar corelația liniară — care măsoară exclusiv tendințe monotone — o ratează complet.

**Această observație constituie ipoteza centrală a lucrării**: dacă relația reală dintre o variabilă și țintă este neliniară, regresia liniară nu o poate reprezenta, în timp ce o rețea neuronală ar trebui să o poată. Ipoteza este verificată în secțiunea 7.3.

---

## 5. Preprocesare și inginerie de caracteristici

### 5.1. Curățarea datelor

Filtrele au fost aplicate secvențial, cu evidența numărului de înregistrări rămase:

| Pas | Înregistrări | Eliminate |
|---|---:|---:|
| Date brute | 30.128 | — |
| Păstrare doar prețuri în EUR | 29.959 | 169 |
| Eliminare valori lipsă esențiale | 29.555 | 404 |
| Suprafață între 15 și 400 m² | 29.243 | 312 |
| Preț între 5.000 și 2.000.000 € | 29.236 | 7 |
| Coordonate în interiorul României | 29.223 | 13 |
| Eliminare valori extreme €/m² | **28.637** | 586 |

Rata de reținere este de **95,0%**.

**Prețurile în lei** (169 de înregistrări, 0,56%) au fost eliminate în locul convertirii. Orice curs de schimb fixat în cod ar fi arbitrar și imposibil de justificat riguros, întrucât anunțurile au fost publicate la date diferite. Pierderea a fost considerată acceptabilă în raport cu introducerea unei valori inventate.

**Filtrarea valorilor extreme s-a făcut pe prețul unitar (€/m²), nu pe preț.** Motivația: un preț de 500.000 € poate fi legitim pentru un apartament mare într-o zonă scumpă, iar unul de 20.000 € pentru o garsonieră într-un oraș mic. Prețul absolut nu permite, singur, distingerea unei erori de o valoare validă. Raportul preț/suprafață normalizează exact factorul relevant. Percentilele 1% și 99% au delimitat intervalul păstrat la **611–3.795 €/m²**.

### 5.2. Caracteristici derivate

**Distanța până la cel mai apropiat centru urban major.** Am definit un set de 10 orașe considerate poli economici (București, Cluj-Napoca, Timișoara, Iași, Constanța, Brașov, Craiova, Galați, Oradea, Sibiu) și am calculat distanța ortodromică prin formula haversine:

$$d = 2R \, \arcsin \sqrt{\sin^2\left(\frac{\Delta\varphi}{2}\right) + \cos\varphi_1 \cos\varphi_2 \sin^2\left(\frac{\Delta\lambda}{2}\right)}$$

unde $R$ este raza Pământului, $\varphi$ latitudinea și $\lambda$ longitudinea, exprimate în radiani.

Variabila rezultată îndeplinește simultan două funcții:

| Localitate | Distanță | Pol cel mai apropiat | Interpretare |
|---|---:|---|---|
| București | 5,5 km | București | centralitate intraurbană |
| Cluj-Napoca | 3,4 km | Cluj-Napoca | centralitate intraurbană |
| **Florești** | **10,9 km** | **Cluj-Napoca** | **proximitate suburbană** |
| Roman | 56,6 km | Iași | periferie |
| Vaslui | 58,6 km | Iași | periferie |

Pentru un apartament din București, distanța măsoară cât de aproape este de centru. Pentru unul din Florești, măsoară cât de aproape este de Cluj. O singură variabilă captează ambele fenomene, inclusiv cazul Florești identificat în analiza exploratorie.

**Codificarea localității prin țintă, cu netezire bayesiană.** Cele 632 de localități au între 1 și 1.445 de anunțuri. Codificarea one-hot ar fi produs sute de coloane aproape goale, iar media brută a unei localități cu două anunțuri ar fi fost zgomot pur.

Am aplicat formula de netezire din secțiunea 2.11, cu $m = 10$ și ierarhie pe două niveluri: media localității se sprijină pe media județului, iar media județului pe media națională.

Exemplu numeric ilustrativ:

| Situație | Calcul | Rezultat |
|---|---|---:|
| Localitate cu 500 de anunțuri, medie 2.100 | $(500 \cdot 2100 + 10 \cdot 1650)/510$ | 2.091 |
| Localitate cu 2 anunțuri, medie 2.800 | $(2 \cdot 2800 + 10 \cdot 1650)/12$ | 1.842 |

Prima își păstrează practic media proprie; a doua este trasă puternic spre valoarea de rezervă.

**Transformări logaritmice.** Suprafața, prețul unitar al localității și distanța au fost logaritmate. Justificarea este dezvoltată în secțiunea 8.1, unde este prezentată problema care a impus această decizie.

### 5.3. Tratarea valorilor lipsă

| Variabilă | Lipsă | Strategie |
|---|---:|---|
| Etaj | 1.427 (5,0%) | Imputare cu mediana + indicator binar |
| An construcție | 7.030 (24,5%) | Imputare cu mediana + indicator binar |

Eliminarea celor 7.030 de înregistrări fără an de construcție ar fi însemnat pierderea unui sfert din setul de date.

Indicatorul binar suplimentar transmite modelului informația că valoarea a fost completată artificial. Raționamentul: **absența unei informații este ea însăși informație**. Un vânzător care omite anul construcției poate avea motive corelate cu prețul. Rezultatele confirmă ipoteza — coeficientul indicatorului de etaj lipsă este −8,94%, al doilea ca magnitudine în modelul liniar.

Medianele de imputare au fost calculate **exclusiv pe setul de antrenare** și aplicate ulterior setului de testare.

### 5.4. Împărțirea setului de date și prevenirea scurgerii de informație

Ordinea operațiilor a fost:

1. Împărțire aleatoare 80/20 → **22.909 antrenare / 5.728 testare**
2. Învățarea codificării localităților **exclusiv pe setul de antrenare** (604 localități prezente)
3. Aplicarea codificării pe ambele seturi
4. Calcularea medianelor de imputare pe antrenare, aplicarea pe ambele

Inversarea pașilor 1 și 2 ar fi produs scurgere de informație: fiecare înregistrare de test și-ar fi contribuit propriul preț la media localității sale, adică la o caracteristică folosită pentru a-i prezice prețul.

Localitățile prezente în test dar absente din antrenare primesc media județului, iar în ultimă instanță media națională.

Sămânța generatorului aleator a fost fixată (`random_state=42`), asigurând identitatea împărțirii între rulări și, implicit, comparabilitatea modelelor.

### 5.5. Matricea finală de caracteristici

Rezultatul procesului este o matrice cu **51 de coloane**:

| Grup | Coloane | Descriere |
|---|---:|---|
| Continue transformate | 3 | log(suprafață), log(€/m² localitate), log(1+distanță) |
| Numerice directe | 3 | camere, etaj, an construcție (ordinal) |
| Binare | 4 | agenție, negociabil, etaj lipsă, an lipsă |
| One-hot județ | 41 | câte o coloană per județ |

**Notă metodologică importantă:** etajul a fost păstrat ca simplă valoare numerică, **fără** construirea unor caracteristici derivate care să codifice explicit forma de U identificată în analiza exploratorie (de exemplu, indicatori de tipul „este ultimul etaj").

Decizia este deliberată. Construirea unor astfel de caracteristici ar fi rezolvat problema *în locul* modelelor, iar regresia liniară ar fi părut artificial mai performantă. Lăsând variabila în formă brută, comparația măsoară exact ceea ce ne interesează: **capacitatea fiecărui model de a descoperi singur o relație neliniară**.

---

## 6. Modelele implementate

### 6.1. Modele de referință

Interpretarea unei valori absolute a erorii necesită puncte de comparație. Am definit două:

**Predicția medianei.** Modelul prezice constant mediana prețurilor din setul de antrenare. Reprezintă pragul minim absolut: orice model care nu îl depășește este inutil. MAE = 38.513 €.

**Regula empirică a evaluatorului.** Modelul prezice produsul dintre prețul mediu pe metru pătrat al localității și suprafața apartamentului — exact raționamentul unui agent imobiliar. MAE = 18.326 €.

Al doilea prag este semnificativ mai exigent: măsoară cât aduce, efectiv, învățarea automată peste bunul-simț cantitativ.

### 6.2. Regresia liniară

Implementare: `scikit-learn`, clasa `LinearRegression`, cu soluție analitică prin metoda celor mai mici pătrate. Am antrenat în paralel și o variantă cu regularizare L2 (`Ridge`, $\alpha = 1$), precedată de standardizare.

Timp de antrenare: sub o secundă.

### 6.3. Rețeaua neuronală — implementarea de referință

Înainte de implementarea în PyTorch, am antrenat aceeași arhitectură folosind `MLPRegressor` din `scikit-learn`.

Scopul nu a fost performanța, ci obținerea unui **reper numeric de verificare**. O implementare scrisă de la zero poate conține erori tăcute — gradienți neresetați, funcție de cost incorectă, model lăsat în modul de antrenare la evaluare — care nu produc excepții, ci doar rezultate slabe. Fără un reper independent, distincția între „modelul nu poate mai mult" și „implementarea mea are o eroare" este imposibilă.

Configurație: două straturi ascunse (64, 32 neuroni), activare ReLU, optimizator Adam, rată de învățare $10^{-3}$, loturi de 256, regularizare L2 $\alpha = 10^{-4}$, oprire timpurie cu răbdare de 15 epoci și fracțiune de validare de 10%.

### 6.4. Rețeaua neuronală în PyTorch

Arhitectura, definită prin moștenire din `nn.Module`:

```
Linear(51 → 64) → ReLU → Linear(64 → 32) → ReLU → Linear(32 → 1)
```

Numărul total de parametri antrenabili: **5.441**.

Stratul de ieșire nu are funcție de activare, întrucât ținta este o valoare continuă nemărginită (logaritmul prețului).

Bucla de antrenare, scrisă explicit, conține cei cinci pași fundamentali:

```python
for batch_X, batch_y in incarcator:
    optimizator.zero_grad()               # 1. sterge gradientii anteriori
    predictii = model(batch_X)            # 2. propagare inainte
    loss = criteriu(predictii, batch_y)   # 3. evaluarea erorii
    loss.backward()                       # 4. propagare inapoi
    optimizator.step()                    # 5. actualizarea parametrilor
```

**Pasul 1** este esențial și adesea omis. PyTorch **acumulează** gradienții în loc să îi înlocuiască. Omiterea instrucțiunii face ca gradientul fiecărui lot să se adune peste toate cele precedente, ducând la pași de actualizare disproporționați și la divergență. Eroarea nu produce nicio excepție.

**Pasul 4** este locul unde se execută propagarea înapoi a erorii. PyTorch construiește dinamic un graf al operațiilor efectuate în propagarea înainte; `backward()` îl parcurge în sens invers, aplicând regula lanțului pentru a obține derivata costului față de fiecare dintre cei 5.441 de parametri.

Mecanisme suplimentare implementate:

- **Comutarea `model.train()` / `model.eval()`** — necesară pentru straturile cu comportament diferit între antrenare și inferență. Arhitectura curentă nu conține astfel de straturi, dar practica previne erori la extinderi ulterioare.
- **Contextul `torch.no_grad()`** la evaluare — dezactivează construirea grafului de calcul, reducând timpul și consumul de memorie.
- **Oprire timpurie cu restaurarea celor mai buni parametri** — parametrii sunt copiați (prin `clone()`) la fiecare îmbunătățire a erorii de validare și restaurați la final. Copierea explicită este obligatorie: `state_dict()` returnează referințe către tensorii modelului, nu copii, astfel încât fără `clone()` „cei mai buni parametri" s-ar modifica odată cu modelul.

Antrenarea s-a încheiat la epoca 80, cel mai bun model fiind obținut la **epoca 65**. Durata: 22 de secunde.

> **Figura 6** — `reports/figures/08_curba_invatare.png`

Curbele de antrenare și validare rămân apropiate pe tot parcursul, fără divergență — semn că oprirea timpurie a prevenit supraînvățarea.

### 6.5. Random Forest

Configurație: 300 de arbori, adâncime nelimitată, minimum 2 exemple per frunză. Timp de antrenare: 10 secunde.

---

## 7. Rezultate și comparație

### 7.1. Performanța comparativă

Toate valorile sunt calculate pe setul de testare (5.728 de anunțuri), neutilizat în nicio etapă anterioară.

| Model | MAE | MedAE | RMSE | MAPE | R² (log) | R² (EUR) |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest (300)** | **14.820 €** | **8.690 €** | **25.448 €** | **14,95%** | **0,850** | **0,812** |
| MLP scikit-learn (64, 64, 32) | 16.106 € | 10.061 € | 25.932 € | 16,44% | 0,833 | 0,804 |
| MLP scikit-learn (64, 32) | 16.144 € | 9.851 € | 26.003 € | 16,21% | 0,833 | 0,803 |
| MLP PyTorch (64, 32) | 16.272 € | 9.950 € | 26.293 € | 16,47% | 0,830 | 0,799 |
| MLP scikit-learn, scalat parțial | 16.633 € | 10.490 € | 26.642 € | 16,97% | 0,819 | 0,794 |
| MLP scikit-learn, scalat integral | 17.140 € | 10.711 € | 27.705 € | 17,34% | 0,812 | 0,777 |
| Regresie liniară | 17.197 € | 10.622 € | 27.921 € | 17,39% | 0,814 | 0,773 |
| Ridge (α = 1) | 17.197 € | 10.618 € | 27.921 € | 17,39% | 0,814 | 0,773 |
| *Referință:* €/m² × suprafață | 18.326 € | 11.905 € | 28.775 € | 19,25% | 0,792 | 0,759 |
| *Referință:* mediana | 38.513 € | 26.500 € | 60.229 € | 43,08% | −0,001 | −0,055 |

### 7.2. Validarea implementării PyTorch

Diferența dintre implementarea proprie și cea de referință este de **0,8%** (16.272 € față de 16.144 €), atribuibilă diferențelor de inițializare aleatoare și de împărțire a setului de validare.

Concordanța confirmă corectitudinea implementării. Utilitatea metodologică a acestui reper merită subliniată: în absența lui, un rezultat de 16.272 € nu ar fi putut fi distins de rezultatul unei implementări cu erori.

### 7.3. Verificarea ipotezei centrale: analiza de sensibilitate

Compararea a două valori MAE indică *care* model este mai bun, dar nu și *ce* a învățat fiecare.

Am dezvoltat, în acest scop, o metodă de **analiză de sensibilitate**: se selectează 300 de apartamente reale din setul de testare, se modifică **o singură caracteristică** la o valoare fixată, se recalculează predicțiile și se raportează media. Repetând pentru fiecare valoare posibilă, se obține răspunsul modelului la acea variabilă, cu celelalte menținute la valorile lor reale.

Rezultatul pentru etaj, exprimat ca variație procentuală față de parter:

| Etaj | Regresie liniară | MLP scikit-learn | MLP PyTorch | Random Forest |
|---:|---:|---:|---:|---:|
| −1 (demisol) | +0,3% | −8,4% | **−15,8%** | −3,0% |
| 0 (parter) | 0,0% | 0,0% | 0,0% | 0,0% |
| 1 | −0,3% | +3,7% | +2,0% | +1,9% |
| 2 | −0,5% | +1,8% | +0,3% | +1,7% |
| 3 | −0,8% | −2,4% | −2,6% | −0,6% |
| **4** | −1,0% | **−5,9%** | **−6,1%** | **−3,2%** |
| 5 | −1,3% | −4,4% | −4,1% | −2,3% |
| 6 | −1,5% | −0,0% | −0,4% | +1,3% |
| 7 | −1,8% | +4,0% | +2,8% | +2,3% |
| **8** | −2,1% | **+6,4%** | **+3,6%** | **+2,6%** |
| 9 | −2,3% | +5,1% | +1,9% | +2,6% |
| 10 | −2,6% | +1,3% | −0,6% | +1,0% |

**Regresia liniară** produce o scădere strict monotonă, de la 0% la −2,6%. Comportamentul este impus de structura modelului: un coeficient unic înmulțit cu valoarea etajului nu poate genera decât o dreaptă. Amplitudinea totală a răspunsului este de 2,9 puncte procentuale.

**Cele trei modele neliniare** identifică, independent unele de altele, același tipar:
- demisolul este cel mai depreciat;
- se produce un **minim local la etajul 4**;
- urmează o **revenire la etajele 7–9**.

Amplitudinea răspunsului rețelei neuronale (de la −15,8% la +6,4%, adică 22,2 puncte procentuale) este de aproximativ **șapte ori** mai mare decât cea a modelului liniar.

Tiparul corespunde exact ipotezei formulate în secțiunea 4.6, pe baza structurii fondului locativ românesc: regimul de înălțime P+4 al blocurilor din perioada comunistă face din etajul 4 ultimul nivel, în timp ce etajele superioare implică o construcție modernă.

**Ipoteza centrală a lucrării este confirmată.** Rețeaua neuronală reconstruiește o relație neliniară pe care regresia liniară nu o poate reprezenta prin construcție, iar reconstrucția corespunde unui fenomen real, verificabil independent. Faptul că trei implementări diferite ajung la același rezultat exclude ipoteza unui artefact al vreunei biblioteci.

### 7.4. Interpretabilitatea modelului liniar

Cu țintă logaritmată, coeficienții se interpretează ca **elasticități**, respectiv ca variații procentuale:

| Variabilă | Coeficient | Efect |
|---|---:|---:|
| log(€/m² localitate) | +1,18797 | elasticitate 1,19 |
| log(suprafață) | +0,86739 | elasticitate 0,87 |
| Etaj lipsă (indicator) | −0,09371 | −8,94% |
| log(1 + distanță) | −0,06283 | −6,09% |
| An construcție (ordinal) | +0,05329 | +5,47% / treaptă |
| Număr camere | +0,02235 | +2,26% / cameră |
| Agenție | −0,01997 | −1,98% |
| Negociabil | −0,01866 | −1,85% |
| An lipsă (indicator) | +0,00450 | +0,45% |
| **Etaj** | **−0,00260** | **−0,26% / etaj** |

**Elasticitatea suprafeței, 0,87**, este subunitară: o creștere de 10% a suprafeței produce o creștere de doar 8,7% a prețului. Apartamentele mari costă, deci, mai puțin pe metru pătrat — un fenomen economic documentat, regăsit aici din date.

**Elasticitatea localității, 1,19**, este supraunitară: piețele scumpe sunt disproporționat mai scumpe decât ar sugera media lor.

**Coeficientul etajului, −0,26% pe etaj**, este practic nul — confirmarea cantitativă a limitării discutate.

Coeficienții variabilelor de județ sunt contraintuitivi (de exemplu, județul Cluj primește un coeficient negativ). Explicația este multicoliniaritatea: variabila `log(€/m² localitate)` conține deja informația privind nivelul de preț al Clujului, astfel încât indicatorul de județ nu mai măsoară nivelul absolut, ci **corecția reziduală**. Coeficienții rămân corecți din punct de vedere matematic, dar **nu pot fi interpretați izolat** — o limitare reală a regresiei liniare cu variabile corelate.

### 7.5. Importanța caracteristicilor în Random Forest

| Caracteristică | Importanță |
|---|---:|
| log(suprafață) | 0,5877 |
| log(€/m² localitate) | 0,2807 |
| log(1 + distanță) | 0,0502 |
| Etaj | 0,0249 |
| An construcție | 0,0229 |
| Toate cele 41 de indicatoare de județ | 0,0168 |
| Număr camere | 0,0065 |

Suprafața și localitatea explică împreună **87%** din capacitatea decizională a modelului.

Importanța redusă a numărului de camere (0,0065) confirmă observația din analiza exploratorie: variabila este redundantă în raport cu suprafața.

Contribuția totală a celor 41 de indicatoare de județ (0,0168) este inferioară celei a etajului singur. Codificarea prin țintă a localității a preluat, practic, întreaga informație geografică relevantă.

### 7.6. Comportamentul erorilor

> **Figura 7** — `reports/figures/06_diagnostic_liniar.png` (regresie liniară)
> **Figura 8** — `reports/figures/07_diagnostic_mlp.png` (rețea neuronală)
> **Figura 9** — `reports/figures/10_diagnostic_rf.png` (Random Forest)

Distribuția erorilor este aproximativ simetrică pentru toate modelele, cu mediana în jurul valorii de −1%. Nu există bias sistematic global.

Reprezentarea erorii procentuale în funcție de prețul real evidențiază însă un tipar comun: **supraestimarea apartamentelor ieftine și subestimarea celor scumpe** — fenomenul cunoscut ca *regresie spre medie*, caracteristic modelelor care subînvață extremele distribuției.

Fenomenul persistă, atenuat, și la modelele neliniare. Explicația nu ține de alegerea modelului, ci de datele disponibile: prețul unei locuințe de lux depinde de calitatea finisajelor, expunere, vecinătate și dotări — variabile absente din setul de date.

### 7.7. Analiza supraînvățării

| Model | MAE antrenare | MAE testare | Diferență | R² antrenare | R² testare |
|---|---:|---:|---:|---:|---:|
| Regresie liniară | 17.024 € | 17.197 € | **+1,0%** | 0,823 | 0,815 |
| MLP PyTorch | 16.009 € | 16.272 € | **+1,6%** | 0,841 | 0,830 |
| MLP scikit-learn | 15.752 € | 16.144 € | +2,5% | 0,848 | 0,833 |
| **Random Forest** | **7.620 €** | **14.820 €** | **+94,5%** | **0,959** | **0,850** |

Rezultatul nuanțează substanțial clasamentul din secțiunea 7.1.

**Random Forest obține cea mai bună performanță pe setul de testare, dar cu prețul unei supraînvățări severe**: eroarea pe datele de antrenare este de aproape două ori mai mică decât cea pe date noi. Modelul memorează în mare măsură setul de antrenare — comportament așteptat pentru arbori cu adâncime nelimitată. Mecanismul de agregare limitează consecințele, dar nu le elimină.

**Rețeaua neuronală prezintă cel mai bun echilibru** între performanță și capacitate de generalizare: diferență de 1,6% între antrenare și testare, la o performanță cu doar 9,8% mai slabă decât Random Forest.

**Regresia liniară nu supraînvață deloc** (+1,0%), dar din motivul opus — capacitatea sa de reprezentare este insuficientă pentru a memora datele. Modelul subînvață.

Constatarea are implicații practice: pe un set de date colectat în altă perioadă sau într-o altă regiune, avantajul Random Forest ar putea să se reducă mai rapid decât cel al rețelei neuronale.

### 7.8. Influența arhitecturii

Testarea unei arhitecturi mai adânci (64, 64, 32) a produs MAE = 16.106 €, față de 16.144 € pentru (64, 32) — o îmbunătățire de **0,24%**, nesemnificativă.

Concluzia: la această dimensiune a setului de date și cu acest număr de caracteristici, **capacitatea rețelei nu constituie factorul limitativ**. Limitarea provine din informația conținută în date. Creșterea complexității arhitecturale nu poate compensa absența unor variabile precum starea finisajelor sau calitatea vecinătății.

---

## 8. Probleme întâmpinate și diagnosticarea lor

Acest capitol documentează trei probleme apărute pe parcurs. Le considerăm partea cea mai instructivă a lucrării: fiecare a produs inițial rezultate greșite fără a genera vreo eroare de execuție, iar identificarea cauzei a necesitat investigație sistematică.

### 8.1. Regresia liniară depășită de o regulă empirică

**Simptom.** La prima evaluare, regresia liniară a obținut rezultate inferioare celor ale modelului de referință empiric:

| Model | MAE | RMSE | R² (EUR) |
|---|---:|---:|---:|
| Referință: €/m² × suprafață | 18.785 € | 30.944 € | 0,741 |
| Regresie liniară | **21.217 €** | **73.263 €** | **−0,451** |

Valoarea R² negativă indica o performanță inferioară predicției mediei — un rezultat inacceptabil pentru un model antrenat.

**Investigație.** Analiza celor mai mari erori a evidențiat un tipar clar:

| Suprafață | Preț real | Preț prezis |
|---:|---:|---:|
| 400 m² | 439.000 € | 3.506.890 € |
| 400 m² | 760.000 € | 3.726.075 € |
| 340 m² | 400.000 € | 1.964.689 € |

Cuantificarea a confirmat concentrarea erorii: **10 înregistrări din 5.730 (0,17%) generau 82,2% din suma pătratelor erorilor.**

**Cauză.** Problema era de natură matematică, nu de implementare. Modelul învăța relația:

$$\log(\text{preț}) = a + b \cdot \text{suprafață}$$

care, prin exponențiere, devine:

$$\text{preț} = e^a \cdot e^{b \cdot \text{suprafață}}$$

Prețul creștea, deci, **exponențial** cu suprafața. Cu $b = 0{,}00948$, un apartament de 60 m² primea factorul $e^{0{,}57} = 1{,}8$, iar unul de 400 m² factorul $e^{3{,}79} = 44$. Pe intervalul uzual (40–80 m²) diferența era neglijabilă; la extreme, predicțiile explodau.

Modelul de referință nu suferea de această problemă, fiind **liniar** în suprafață prin construcție.

**Soluție.** Logaritmarea și a variabilei de intrare transformă relația în:

$$\log(\text{preț}) = a + b \cdot \log(\text{suprafață}) \quad \Longrightarrow \quad \text{preț} = e^a \cdot \text{suprafață}^{\,b}$$

adică o **lege de putere**. Cu $b \approx 1$ se obține proporționalitate, comportamentul corect pentru bunuri imobiliare. Același tratament a fost aplicat prețului unitar al localității și distanței, ambele acționând multiplicativ.

**Rezultat.**

| Indicator | Înainte | După | Ameliorare |
|---|---:|---:|---:|
| MAE | 21.217 € | 17.197 € | −19,0% |
| RMSE | 73.263 € | 27.921 € | **−61,9%** |
| R² (EUR) | −0,451 | +0,773 | — |

**Concluzie metodologică.** Alegerea formei funcționale este o decizie de modelare, nu un detaliu de implementare. Un model corect implementat, dar cu formă funcțională inadecvată, poate fi depășit de o regulă empirică simplă. Includerea unui model de referință în protocolul de evaluare a fost esențială: fără el, valoarea de 21.217 € ar fi părut acceptabilă.

### 8.2. Standardizarea care deteriorează performanța rețelei

**Simptom.** Recomandarea standard din literatură este ca datele de intrare ale unei rețele neuronale să fie standardizate. Aplicarea ei a produs însă rezultate inferioare:

| Sămânță aleatoare | Cu standardizare | Fără standardizare |
|---:|---:|---:|
| 0 | 16.973 € | 16.160 € |
| 1 | 17.213 € | 16.292 € |
| 42 | 17.140 € | 16.144 € |
| 7 | 17.268 € | 16.232 € |

Rezultatul este consistent pe toate rulările, deci nu este atribuibil variabilității aleatoare.

**Investigație.** Standardizarea împarte fiecare coloană la deviația ei standard. Pentru variabilele indicatoare de județ, cu foarte puține valori nenule, deviația standard este mică, iar factorul de amplificare este mare:

| Element | Deviație standard | Factor de amplificare |
|---|---:|---:|
| Indicator județ Harghita | 0,0288 | **34,7×** |
| Indicator județ cel mai frecvent | 0,242 | 4,1× |
| log(€/m² localitate) | 0,25 | 4,1× |

**12 din cele 41 de indicatoare de județ sunt amplificate de peste 10 ori.** Cea mai puțin frecventă variabilă indicatoare primește o pondere inițială de opt ori mai mare decât cea mai predictivă caracteristică din întregul set.

Ipoteza a fost testată prin standardizarea selectivă, aplicată exclusiv celor 10 caracteristici continue:

| Strategie | MAE (medie pe 4 rulări) |
|---|---:|
| Standardizare integrală | 17.149 € |
| Standardizare doar a variabilelor continue | 16.691 € |
| Fără standardizare | **16.207 €** |

Standardizarea selectivă recuperează aproximativ jumătate din diferență, confirmând parțial ipoteza.

**Cauza celei de-a doua jumătăți** rezidă în transformările aplicate anterior. După logaritmare, caracteristicile continue se află deja pe scări comparabile:

| Caracteristică | Medie | Deviație standard |
|---|---:|---:|
| log(€/m² localitate) | 7,39 | 0,25 |
| log(suprafață) | 4,01 | 0,42 |
| log(1 + distanță) | 3,00 | 1,46 |
| Număr camere | 2,40 | 0,91 |
| Etaj | 2,74 | 2,20 |
| An construcție | 1,52 | 1,05 |

Toate valorile se încadrează în același ordin de mărime. **Standardizarea nu mai are ce să corecteze, dar efectele ei secundare persistă.**

**Concluzie metodologică.** Regula generală rămâne validă: standardizarea este esențială când caracteristicile diferă cu ordine de mărime. În cazul de față, transformările logaritmice aplicate în etapa de inginerie a caracteristicilor rezolvaseră deja problema. Recomandările standard trebuie verificate empiric în contextul concret, nu aplicate mecanic. Variabilele codificate one-hot, în particular, nu ar trebui standardizate.

### 8.3. Coordonate geografice invalide

**Simptom.** Distanța maximă calculată până la un centru urban era de **7.951 km** — imposibilă pentru un anunț din România.

**Investigație.** Nouă anunțuri prezentau coordonate în afara teritoriului național:

| Localitate declarată | Latitudine | Longitudine | Locație reală |
|---|---:|---:|---|
| Ciupercenii Noi (Dolj) | −0,0002 | 0,0002 | Golful Guineei |
| Sulina (Tulcea) | 16,06 | 108,24 | Vietnam |
| Beba Veche (Timiș) | 51,56 | 7,09 | Germania |
| Sânislău (Satu Mare) | 53,59 | 10,05 | Germania |

Coordonatele apropiate de (0, 0) corespund unei erori frecvente în sistemele informatice: câmpuri necompletate inițializate cu zero. Celelalte provin probabil din erori de introducere manuală.

**Impact.** Deși reprezintă doar 0,03% din date, aceste înregistrări afectau disproporționat: distanțele de mii de kilometri deveneau valori extreme care distorsionau atât normalizarea, cât și învățarea variabilei de distanță.

**Soluție.** Filtrarea prin verificarea încadrării în dreptunghiul geografic al României:

```python
in_tara = (df["lat"].between(43.5, 48.5)) & (df["lon"].between(20.0, 30.0))
```

**Concluzie metodologică.** Datele provenite din formulare completate de utilizatori conțin erori sistematice. Validarea prin verificarea încadrării în intervale plauzibile trebuie aplicată tuturor variabilelor cu domeniu cunoscut.

### 8.4. Sinteză

Cele trei probleme au două trăsături comune.

**Niciuna nu a generat o eroare de execuție.** Programul a rulat complet în toate cazurile, producând rezultate plauzibile la prima vedere.

**Toate au fost detectate prin instrumente de diagnostic, nu prin metrici agregate.** Metrica MAE indica un rezultat mediocru, dar nu și cauza. Identificarea a necesitat: analiza celor mai mari erori individuale, testarea sistematică pe mai multe semințe aleatoare, examinarea valorilor extreme ale variabilelor derivate.

Concluzia practică: **un protocol de evaluare bazat exclusiv pe metrici agregate este insuficient.** Modelele de referință, reprezentările grafice de diagnostic și analiza distribuției erorilor nu sunt elemente accesorii, ci instrumente indispensabile de depanare.

---

## 9. Concluzii

### 9.1. Rezultate obținute

Lucrarea a realizat un sistem complet de predicție a prețului locuințelor, de la colectarea datelor până la modele antrenate și evaluate.

**Setul de date** cuprinde 28.637 de anunțuri validate, din 632 de localități și toate cele 41 de județe, colectate direct din piața reală prin interfața publică a platformei OLX.

**Cele mai bune rezultate**, exprimate în eroare procentuală absolută medie: 14,95% pentru Random Forest, 16,47% pentru rețeaua neuronală implementată în PyTorch și 17,39% pentru regresia liniară. Toate cele trei depășesc modelul de referință empiric (19,25%) și, cu mult, predicția medianei (43,08%).

Pe o piață în care negocierea uzuală reprezintă 5–10% din preț, o eroare medie de circa 15% este un rezultat rezonabil pentru un model construit exclusiv pe caracteristici structurale.

### 9.2. Răspunsul la întrebarea comparativă

Obiectivul principal al lucrării a fost compararea regresiei liniare cu o rețea neuronală. Răspunsul are două componente.

**Cantitativ**, rețeaua neuronală depășește regresia liniară cu 5,4% la eroarea absolută medie (16.272 € față de 17.197 €) și cu 0,92 puncte procentuale la eroarea relativă. Diferența este reală, dar modestă.

**Calitativ**, diferența este de natură structurală. Analiza de sensibilitate demonstrează că rețeaua reconstruiește relația neliniară dintre etaj și preț — minim la etajul 4, revenire la etajele superioare — cu o amplitudine de aproximativ șapte ori mai mare decât cea a modelului liniar, care poate produce doar o tendință monotonă.

Tiparul identificat corespunde unei realități verificabile: regimul de înălțime P+4 al blocurilor construite în perioada 1960–1990. Modelul a extras această regularitate exclusiv din date, fără nicio informație prealabilă despre arhitectura fondului locativ.

**Concluzia comparației** este că avantajul rețelei neuronale nu constă în precizia globală, ci în capacitatea de a reprezenta relații pe care modelul liniar nu le poate exprima prin construcție. Într-un context în care se urmărește înțelegerea mecanismelor pieței, nu doar minimizarea erorii, această distincție este determinantă.

### 9.3. Observația privind Random Forest

Includerea unui al treilea model a produs rezultatul cel mai contraintuitiv al lucrării: **Random Forest depășește rețeaua neuronală cu 8,9%**, antrenându-se în 10 secunde, fără preprocesare specifică și fără ajustarea hiperparametrilor.

Rezultatul confirmă observația frecventă din literatură (Grinsztajn et al., 2022) conform căreia metodele bazate pe arbori rămân competitive sau superioare pe date tabulare de dimensiuni moderate.

Analiza supraînvățării nuanțează însă acest avantaj. Random Forest prezintă o diferență de 94,5% între eroarea de antrenare și cea de testare, față de 1,6% pentru rețeaua neuronală. Modelul memorează în mare măsură setul de antrenare, iar avantajul său ar putea fi mai puțin stabil pe date colectate în alte condiții.

### 9.4. Limitări

**Prețul cerut nu este prețul de tranzacționare.** Modelul prezice valoarea din anunț, care în practică se negociază cu 5–10%. Datele privind tranzacțiile efective nu sunt public accesibile.

**Setul de date reprezintă o fotografie instantanee** a zilei de 20 august 2026. Anunțurile apar și dispar continuu; o nouă colectare nu ar reproduce identic același set.

**Acoperirea este parțială.** Limita de paginare de 1.000 de rezultate per interogare face ca județele cu volum mare — București, Cluj — să rămână plafonate chiar și după partiționare.

**Variabile absente.** Setul nu conține informații privind starea finisajelor, expunerea, calitatea vecinătății, prezența liftului, existența unui loc de parcare sau eficiența energetică. Analiza erorilor sugerează că acestea explică o parte semnificativă din varianța rămasă neexplicată.

**Structura sursei.** 83,5% dintre anunțuri provin de la agenții imobiliare, ceea ce poate introduce o distorsiune sistematică față de piața totală.

**Granularitatea anului de construcție.** Variabila este disponibilă doar ca interval, nu ca an exact, ceea ce limitează precizia estimării vechimii.

### 9.5. Direcții de dezvoltare

**Îmbogățirea setului de caracteristici** prin extragerea de informații din textul descrierilor anunțurilor (prezența liftului, starea finisajelor, existența unui balcon) folosind procesare de limbaj natural.

**Modelarea temporală** prin colectări repetate la intervale regulate, permițând analiza evoluției prețurilor și introducerea unei componente sezoniere.

**Gradient boosting** (XGBoost, LightGBM) ca alternativă la Random Forest, cu ajustarea sistematică a hiperparametrilor prin validare încrucișată.

**Estimarea incertitudinii**, prin regresie cuantilică sau ansambluri, pentru a furniza intervale de încredere în locul unei valori punctuale — mult mai util în practică.

**Modele specializate pe segmente**, dat fiind că analiza erorilor sugerează comportamente distincte între piața de masă și segmentul premium.

### 9.6. Concluzie generală

Lucrarea a demonstrat că un model de predicție a prețurilor imobiliare poate fi construit integral pe date colectate independent, atingând o eroare medie de aproximativ 15% pe piața românească.

Rezultatul metodologic principal nu este însă o cifră, ci o observație despre practica modelării: în toate cele trei probleme majore întâmpinate, cauza nu a fost algoritmul, ci **decizii referitoare la date** — forma funcțională a variabilelor, preprocesarea aplicată, validarea valorilor de intrare. Niciuna nu ar fi fost identificată prin simpla urmărire a metricii de performanță.

---

## 10. Bibliografie

1. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
3. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
5. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems*, 35.
6. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (ed. a 2-a). Springer.
7. Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257.
8. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations (ICLR)*.
9. Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection: A Practical Approach for Predictive Models*. CRC Press.
10. Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations Newsletter*, 3(1), 27–32.
11. Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, 32.
12. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
13. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.

---

## 11. Anexe

### Anexa A — Structura proiectului

```
├── run_scraper.py              colectarea datelor de pe OLX
├── train_all.py                pipeline-ul complet
├── src/
│   ├── scraper/
│   │   ├── fetch.py            cereri HTTP, paginare, reincercari
│   │   ├── storage.py          salvare / incarcare JSON
│   │   └── regions.py          identificatori judete si categorii
│   ├── data/
│   │   ├── build_dataset.py    curatare si filtrare -> CSV
│   │   └── eda.py              analiza exploratorie
│   ├── features/
│   │   └── build_features.py   inginerie de caracteristici, impartire, encoder
│   └── models/
│       ├── evaluate.py         metrici, tabel comparativ, diagnostic
│       ├── persist.py          salvarea modelelor antrenate
│       ├── linear.py           regresie liniara + Ridge + referinte
│       ├── mlp_sklearn.py      MLPRegressor (reper de verificare)
│       ├── mlp_torch.py        reteaua in PyTorch
│       └── random_forest.py    Random Forest
├── notebooks/                  explorare interactiva
├── data/raw/                   date brute (imutabile)
├── data/processed/             date curatate, antrenare / testare, encoder
├── models/                     modele antrenate
└── reports/figures/            reprezentari grafice
```

### Anexa B — Reproducerea rezultatelor

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

python run_scraper.py      # colectarea datelor (~22 min, optional)
python train_all.py        # pipeline complet (~70 s)
```

### Anexa C — Mediul de lucru

| Componentă | Versiune |
|---|---|
| Python | 3.11.9 |
| pandas | 3.0.5 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.0 |
| PyTorch | 2.13.0 (CPU) |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| Sistem de operare | Windows 11 |

### Anexa D — Lista figurilor

| Nr. | Fișier | Conținut |
|---:|---|---|
| 1 | `01_distributie_pret.png` | Distribuția prețului, brut și logaritmat |
| 2 | `02_pret_vs_suprafata.png` | Relația preț–suprafață, scară liniară și dublu-logaritmică |
| 3 | `03_pret_per_mp_judete.png` | Distribuția prețului unitar pe județe |
| 4 | `04_corelatii.png` | Matricea de corelații |
| 5 | `05_factori.png` | Efectul camerelor, etajului și perioadei de construcție |
| 6 | `08_curba_invatare.png` | Curba de învățare a rețelei PyTorch |
| 7 | `06_diagnostic_liniar.png` | Diagnosticul erorilor — regresie liniară |
| 8 | `07_diagnostic_mlp.png` | Diagnosticul erorilor — rețea neuronală |
| 9 | `10_diagnostic_rf.png` | Diagnosticul erorilor — Random Forest |

### Anexa E — Tabelul complet al rezultatelor

Disponibil în format prelucrabil la `reports/rezultate_modele.csv`, regenerat la fiecare rulare a pipeline-ului.
