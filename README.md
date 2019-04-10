# Descriptif du projet
***

Pour notre projet fil rouge, nous allons travailler sur la classification (`text mining`) d'offre d'emploi au Etat Unis.
Tout d'abord, nous allons diviser notre travail en trois parties:

* Scrapper les offres d'emploi sur le site "indeed" sur les différents types de poste suivant:
  * **DATA SCIENCE**
  * **DEVELOPPER**
  * **ACTUARIAT**
  * **DIGITAL MARKETING**
  (Autres variables ajouter récements)

* Nettoyage  et analyse descriptives des données retenus

* Traitement des données proprement dites.



***
## SCRAP
Nous avons scrapper les offres dans différentes villes au USA pour les métiers cité ci-haut. Le but est de bien spécifier l'intitulé de l'offre, l'entreprise ayant publier l'offre, la localisation, le résumé du poste et la description du poste. Le lien vers les différents fichier se trouve à l'adresse suivante : `http....`

## Nettoyage et analyse descriptive des données
Tous les fichiers sont de la même forme c'est à dire contient les même variables comme décrit dans la partie scrap.
Nous avons X lignes pour **DATA SCIENCE**, X lignes **DEVELOPPER**, X lignes pour **ACTUARIAT** et en fin X lignes pour **DIGITAL MARKETING**
Pour ces différents postes, nous avons fait:
  * concatenation des données et rajout d'une colonne `labels`
  * filtrage sur les postes réellement appropriés par la variables intitulé du poste
  * suppression des ponctuations
  * transformer tout le text en minuscule
  * construction du sac de mot (bag of word)
  * visualisation de l'occurence des mots les plus fréquent par un diagramme en bar et par le wordcloud
  * définition des outils (logiciels ou applications) et visualisation ce ceux les plus important par un diagramme en bar et wordcloud

> Un exemple des occurrences des outils les plus utilisés dans le cas de la data science
![](https://github.com/Tantelitiana22/text_mining_fil_rouge/blob/master/image_file/word_clue_hist.png)

> Un exemple des mots les plus récurrents dans la data science
![](https://github.com/Tantelitiana22/text_mining_fil_rouge/blob/master/image_file/word_cloud.png)

## Traitement des données proprement dites
### Normalisation des données
  * suppression des unicodes, urls et stopword
  * lemmatisation
  * présentation du TF-IDF
    * TF

    TF ou term frequency:
Le TF consisite tout simplement à calculer le nombre d'occurence d'un terme dans un document, soit la fréquence. On définit le TF comme suite:
Soit P l'ensemble des lettres qui se trouvent dans notre corpus. Soit i l'indice d'une lettre se trouvent dans P et j l'indice d'un document dans notre corpus.

    * IDF

La fréquence inverse de document (inverse document frequency) est une mesure de l'importance du terme dans l'ensemble du corpus.Dans le schéma TF-IDF, elle vise à donner un poids plus important aux termes les moins fréquents, considérés comme plus discriminants. Elle consiste à calculer le logarithme (en base 10 ou en base 21) de l'inverse de la proportion de documents du corpus qui contiennent le terme : $$IDF_{i,j}=log\left( \frac{|D|}{|d_j,t_i \in d_j|}\right)+1$$
$|D|$: Nombre total de documents dans le corpus.
$|d_j,t_i \in d_j|$: Nombre de document où apparait le mot $t_i$ dans le corpus.

    * TF-IDF

### Modelisation
  * NAÏF BAYES
  * SVM
  * LOGISTIQUE
## Transformation avec des techniques de wordembeding :
