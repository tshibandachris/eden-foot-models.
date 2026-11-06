
# VUE D’ENSEMBLE DE L’ARCHITECTURE

> **Statut :** Document d’architecture de référence  
> **Portée :** Documentation commune destinée aux dépôts `eden-foot-models` et `eden-engine`  
> **Version :** 1.0

## 1. Objet

Ce document décrit la séparation organisationnelle et conceptuelle entre les deux dépôts GitHub suivants :

- `tshibandachris/eden-foot-models`
- `tshibandachris/eden-engine`

Il établit leur périmètre respectif, les principes de gouvernance documentaire et les règles générales qui encadrent toute évolution de leur relation.

Ce document constitue une référence d’architecture. Il ne remplace ni les spécifications propres à chaque dépôt, ni les décisions techniques approuvées, ni les procédures opérationnelles qui y sont documentées.

## 2. Principes directeurs

L’organisation des deux dépôts repose sur les principes suivants.

### 2.1 Séparation des responsabilités

Chaque dépôt possède un objectif, un périmètre et un cycle de changement distincts. Une modification dans l’un ne doit pas être considérée comme une modification implicite de l’autre.

### 2.2 Interfaces explicites

Aucune dépendance entre les dépôts ne doit être supposée. Toute interface éventuelle doit faire l’objet d’une décision distincte, documentée et approuvée avant d’être considérée comme une référence commune.

### 2.3 Traçabilité

Les décisions importantes, les changements de périmètre et les validations doivent être consignés dans les documents ou les mécanismes de suivi appropriés.

### 2.4 Reproductibilité documentaire

Les documents de référence doivent préciser leur portée, leurs hypothèses, leurs limites et, lorsque cela est pertinent, les versions auxquelles ils s’appliquent.

### 2.5 Modifications maîtrisées

Les changements doivent être examinés dans le dépôt concerné. Une validation obtenue dans un dépôt ne constitue pas automatiquement une validation de l’autre.

### 2.6 Gouvernance indépendante

Chaque dépôt conserve ses propres règles de contribution, ses contrôles, ses validations et son historique.

## 3. Vue conceptuelle

```text
                    ÉCOSYSTÈME EDEN MATCHLAB
                              |
                +-------------+-------------+
                |                           |
                v                           v
       eden-foot-models               eden-engine
       Dépôt indépendant              Dépôt indépendant
                |                           |
       Périmètre de recherche        Périmètre applicatif
       et documentation              et documentation
                |                           |
       Historique et contrôles       Historique et contrôles
       propres au dépôt              propres au dépôt
                |                           |
                +-------------+-------------+
                              |
                 Toute interface éventuelle
                 exige une décision explicite
```

Ce schéma représente une séparation de responsabilités. Il ne décrit pas une intégration existante et ne doit pas être interprété comme la preuve d’un échange de données ou d’un couplage logiciel entre les dépôts.

## 4. Périmètre du dépôt `eden-foot-models`

Le dépôt `tshibandachris/eden-foot-models` constitue un espace indépendant consacré aux travaux de recherche et à leur documentation.

Sa documentation peut notamment préciser :

- l’objectif et le périmètre des travaux ;
- les définitions, hypothèses et conventions retenues ;
- la provenance et les limites des données mentionnées ;
- les méthodes de validation documentaire et scientifique ;
- les versions des documents et les décisions associées ;
- les limites connues et les réserves d’interprétation.

Les résultats ou conclusions présentés dans ce dépôt doivent être accompagnés d’un contexte suffisant pour éviter toute interprétation dépassant leur portée réelle.

Ce document ne certifie aucun résultat, aucune méthode et aucune performance.

## 5. Périmètre du dépôt `eden-engine`

Le dépôt `tshibandachris/eden-engine` constitue un espace indépendant consacré au périmètre applicatif et à sa documentation.

Sa documentation peut notamment préciser :

- l’objectif et les limites du logiciel ;
- l’organisation générale des composants documentés ;
- les responsabilités des interfaces explicitement approuvées ;
- les conventions de stockage et de gestion des données ;
- les exigences générales de maintenance ;
- les procédures de validation et de publication propres au dépôt.

Toute description détaillée d’un composant doit être vérifiée dans les fichiers et les versions réellement présents dans ce dépôt.

Ce document ne prétend pas décrire des composants qui n’auraient pas été inspectés.

## 6. Frontières entre les dépôts

Les règles suivantes s’appliquent à toute évolution de l’organisation :

1. Les deux dépôts conservent des historiques Git distincts.
2. Une modification dans un dépôt ne modifie pas automatiquement l’autre.
3. Des noms de fichiers identiques ne signifient pas que leur contenu ou leur version sont automatiquement synchronisés.
4. Toute interface éventuelle doit être décrite dans une spécification dédiée.
5. Toute décision de partage doit préciser les responsabilités, les versions concernées et les modalités de validation.
6. En l’absence d’une décision explicite, les dépôts sont considérés comme indépendants.

## 7. Gouvernance GitHub

Les pratiques suivantes constituent des recommandations de gouvernance documentaire. Elles doivent être adaptées aux règles effectivement configurées dans chaque dépôt.

### 7.1 Branche par défaut

La branche par défaut doit être protégée selon les possibilités et les règles du dépôt. Les changements importants devraient passer par une Pull Request (PR) plutôt que par une modification directe non examinée.

### 7.2 Branches de travail

Les branches de fonctionnalité, de correction ou de maintenance doivent avoir un objectif identifiable et un périmètre limité.

### 7.3 Pull Requests

Une PR devrait présenter :

- le motif du changement ;
- les fichiers concernés ;
- les conséquences attendues ;
- les vérifications effectuées ;
- les limites ou points restant à examiner.

### 7.4 Intégration continue

Les contrôles CI doivent être définis et interprétés séparément pour chaque dépôt. Le succès d’un pipeline ne vaut que pour le commit et le dépôt auxquels il se rapporte.

Aucun résultat CI n’est déclaré dans le présent document.

## 8. Documentation et éléments de preuve

Les éléments documentaires doivent permettre de distinguer clairement :

- une décision approuvée ;
- une proposition en cours d’examen ;
- une observation issue d’une vérification ;
- un résultat de test ;
- une hypothèse non vérifiée.

Les comptes rendus de validation devraient identifier le dépôt, la branche, le commit et le périmètre effectivement examinés lorsque ces informations sont disponibles.

Une affirmation de validation ne doit pas être étendue à des éléments qui n’ont pas été contrôlés.

## 9. Liste de contrôle pour l’examen d’un changement

Avant d’approuver un changement documentaire ou organisationnel, vérifier les points suivants :

- [ ] Le dépôt concerné est clairement identifié.
- [ ] Le changement reste dans le périmètre de ce dépôt.
- [ ] Les hypothèses et les limites sont explicites.
- [ ] Les éventuelles dépendances sont documentées plutôt que supposées.
- [ ] Les références de version et de commit sont exactes lorsqu’elles sont mentionnées.
- [ ] Les validations annoncées correspondent à des contrôles réellement effectués.
- [ ] Le contenu ne présente pas une proposition comme une décision déjà approuvée.
- [ ] Les conséquences sur l’autre dépôt sont examinées séparément si nécessaire.

## 10. Limites de ce document

Ce document :

- ne constitue pas un audit du contenu actuel des deux dépôts ;
- ne certifie pas leur état de sécurité ;
- ne certifie pas l’exactitude mathématique ou scientifique de travaux éventuels ;
- ne déclare aucun test, pipeline CI ou déploiement réussi ;
- ne crée aucune interface technique entre les dépôts ;
- ne remplace pas les fichiers de conception, spécifications ou procédures propres à chaque dépôt.

Les descriptions des dépôts doivent être précisées à partir de leur contenu réel et de leurs versions vérifiées.

## 11. Conditions d’archivage dans les deux dépôts

Ce fichier est destiné à être placé à la racine de chacun des dépôts :

- `tshibandachris/eden-foot-models/ARCHITECTURE_OVERVIEW.md`
- `tshibandachris/eden-engine/ARCHITECTURE_OVERVIEW.md`

Avant publication :

1. Examiner le contenu dans le contexte de chaque dépôt.
2. Vérifier qu’aucune phrase ne laisse entendre qu’une intégration non approuvée existe.
3. Conserver le même texte de référence dans les deux dépôts, sauf décision documentée justifiant une adaptation.
4. Publier les changements séparément et vérifier chaque commit.
5. Consigner les éventuelles différences de version si les deux publications ne sont pas effectuées simultanément.

## 12. Historique documentaire

| Version | Description |
|---|---|
| 1.0 | Création de la vue d’ensemble architecturale commune en français. |

---

**Fin du document.**
