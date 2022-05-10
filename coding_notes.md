- Max pooling might be a mistake in my model. Max pooling will remove the frequency aspect, where if a user reads 
  articles on the following locations \[Trondheim, Oslo, Trondheim, Trondheim, Trondheim\], this will essentially be the 
  same as \[Trondheim, Oslo\]. 
- Average Pooling would fix this but has the issue of reducing the most outlying values in the embedding. Such as Trysil
  should have a large value in a col somewhat describing winter sports or something like that, but averages it down by
  the many more instances of the user's hometown.

- How do I Train/Test split this data? Should I take out 20% of users and have a clean run on these users? 
  - or split on interactions? This will allow the model to train on the same users but on different interactions.
  - I think the user-split is more intuitive and makes more sense to train on some users and then test performance on 
    clean users as this is closer to the actual task. Will probably give a better gauge of the actual goodness.
  - Heri: ta det på brukere

- News locations are several per news. The location-module must take a variable number of ints and embed each of these.
  - History location is only one per news, but should make the adaptive and use for them as well to keep fairness.

- country position data is given in alpha-2 which must be converted to country names to be valid for embedding.
  - Need to convert to name. But name may be several words. Must either choose most significant name manually: 
    - If i create this list as a file and add to my project i can add this to "Contributions". 
  - Source of error: Embedder only takes a single word, although several countries, such as costa rica, have multiple. 
    Selection of which word to use was done subjectively by me. "El Salvador" should obviously become salvador and not "el".
  - This may be happening in the location-parser as well, need to check this!
  
- Is attention order-agnostic?  


Dataflow:

_______________________________________
Baseline:

single article:
 |Title|Body|cat|subcat|
- len: title_size + body_size + 2

Single Interaction
- |Candidate article data|history_article_data|

- len: (title_size + body_size + 2) * (1+hist_size) 
  - One for candidate, rest for history
  
_______________________________________
With Position:

single article:
 |Title|Body|cat|subcat|Position|
- len: title_size + body_size + 3

Single Interaction
- |Candidate article data|history_article_data|

- len: (title_size + body_size + 3) * (1 + hist_size) 
  - One for candidate, rest for history

_______________________________________
With Location:

single article:
 |Title|Body|cat|subcat|Location|
- len: title_size + body_size + 2 + max_location_size

Single Interaction
- |Candidate article data|history_article_data|

- len: (title_size + body_size + 2 + max_location_size) * (1+hist_size) 
  - One for candidate, rest for history


_______________________________________
+---------------------------------------+
|DATA INPUT:
+---------------------------------------+
| X: (candidate, history)
|  - cand shape: (batch_size, title+body+2)
|  - hist shape: (batch_size, history_size, title+body+2)




















