- [ ] Padde/forkorte tittel og body før den skal inn i embeddern.
  - Husk at embeddern ikke har noe \<pad\> token som det er no. Kan evt bare padde med \[np.Zeroes(n-k, 300)\]

- [ ] Må finne en vettug min_count. En lav min_count gir færre "Title = []" problemer.
  - min_count = 1 gir title-empty: 2.26%, body_empty: 2.71%, both_empty: 0.25% med vocab = 636 657
  - min_count = 10 gir title-empty: 3.27%, body_empty: 2.71%, both_empty: 0.25% med vocab = 86 361
  - min_count = 25 gir title-empty: 3.84%, body_empty: 2.71%, both_empty: 0.25% med vocab = 45 495
  - min_count = 50 gir title-empty: 4.60%, body_empty: 2.71%, both_empty: 0.25% med vocab = 27 447
  - min_count = 100 gir title-empty: 5.71%, body_empty: 2.71%, both_empty: 0.25% med vocab = 16 476
  - min_count = 500 gir title-empty: 11.92%, body_empty: 2.71%, both_empty: 0.25% med vocab = 4 668
  - min_count = 1000 gir title-empty: 17.46%, body_empty: 2.71%, both_empty: 0.25% med vocab = 2 469
FASIT:
- ord med får counts er uansett ubrukelig.
- se på zipfs law for kutting av mest brukte.
- ngram: slå sammen prevalente ord. Se på to og to ord som dukker opp samtidig.
- "Lete etter ord som gjør setningene unike, uten å gjøre det for unikt."

- [ ] Hvis jeg ikke har noe intensjon om å fortsette å trene embedderne kan jeg trekke uk kun KeyedVectors 
objektet (embedder.wv). Den er MYE mindre. Jeg bruker uansett ikke modellen senere. Jeg laster embeddern rett fra 
numpy-filen. Sjekk at denne er KeyedVectors objektet, før jeg slette noe. 

- [ ] It is VERY slow to iterate over interactions and adding a column for article_id. Should rather do this on the fly
  when generating data in generator.
  - So whenm generating data, match canon_url with manifest concurrently, rather than doing it upfront.
- [X] Change generator to Yielder rather than returner.
- [ ] Add historical user location to generator for special user position encoder.
- [ ] Make get_article_id_encoded_from_url return ints by setting the"if not in" to -1 rather than None, then instead of dropna, set some other function which drops -1.
  - Maybe dropna has a possibility of setting drop_values and including -1.
- [ ] Change location representation type to int. It is currently Float to allow for None's.

- [ ] Right now the Zeroes representing nothingness actually gives the index of the word "sier".

GENERATOR -> Full batch:
- Need to predetermine the size of the final batch:
  - len(interactions) - 3*users (first 3 is always lost to history)


Implement metrics:
- Ndcg: sklearn.metrics.ndcg_score(y, y_pred)
  - Requires on shape (no_samples, n_labels), but in reality i must 


Create better test dataset. It is currently too easy. To possibilities:
  - Fresh News: only sample negs from articles PUBLISHED on the day in question.
  - Read News: only sample negs from articles READ on the day in question.

Implement WandB on IDUN. Generate some data, 