

Part of Multi View:

News            -> nenc w att -> news encoded
(t|b|c|sc|loc)  -> nenc w att -> (r_n)

Then use as normal

- Might overlap too closely with the words in the text, no impact
- What would one then do with user position?

__________________________________
Taken directly out of news:

Cand:

News loc        -> pad                         -> embed                          -> max_pool
(l_1, ..., l_k) -> (l_1, ..., l_k, 0, ..., 0)  -> (e_1, ..., e_k, e_0, ..., e_0) -> (avg(e))
(k)                (max_locations_per_article) -> (max_loc, embed_size)          -> (embed_size)

- Real
start with 1 news with k locations, already padded to l locations: (1,l)
embed with embed_size e: (1, l) -> (1, l, e)
flatten: (1, l, e) -> (l, e)
avg pooling: (l, e) -> (e)


hist:

News loc              -> pad                         -> embed                          -> max_pool
(locs_n1, locs_nk)   -> (locs_padded_1, ..., lp_k, 0_locs_max_hist)  -> (e_1, ..., e_k, e_0, ..., e_0) -> (avg(e))
(k, locs_per_article) -> (max_locations_per_article) -> (max_loc, embed_size)          -> (embed_size)

start with n news with l locations per (already padded) : (h, l)
embed with embed_size e: (h, l) -> (h, l, e)
reshape: (h, l, e) -> (h*l, e)
avg pooling: (h*l, e) -> e

h fixed, l fixed, e fixed,


The combination of loc score and news score can be implemented as pure dot or neural. Neural is supported by the XOR 
principle. This can be cited. The guy in life 3.0 cited something a student of his proved about length vs depth. Okura 
states that dot is best in the combination of embeddings, however in combining scores the dot product is much worse. 
This because if we have a full match on news interests but no match on location it will not be recommended, and also 
if full match on location, but no match on news. This is the XOR principle. The inner(dot) product favors the news which 
are mediocre fits in both categories over full matches on either or. Could try to implement a sum operation of some sort. 