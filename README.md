# Q-A_Covid
## Attention le code telecharge automatiquement les poids des modeles avant (~4 x 800Mo)
### Pour rouler la demo : 
`cd demo_q_a`
`python craft_files.py` Pour calculer les embeddings necessaires
`python Demo_Q_A.py`
La demo se lance avec quelques questions deja faites puis lance le mode interactif

### Pour continuer sur la prod
#### Client
Le client (en ts) se trouve dans le dossier client.
A date un test simple a ete fait, il faut maintenant reproduire le comportement du fichier craft_files.py dans demo_q_a
* Prendre toutes les datas interessantes et calculer / stocker les embeddings  
* Faire une cosine / euclidian distance (ou plus fancy) pour trouver les documents les plus proches 

#### Server
Le serveur se trouve dans le dossier serveur, il faut avoir un terminal ouvert a cote et : 
```python server/serveur.py```
Puis attendre le `Modele loaded`