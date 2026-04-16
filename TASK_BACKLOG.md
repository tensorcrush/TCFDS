# Backlog de tâches (audit ciblé)

## 1) Coquille typographique / incohérence de version
- **Constat** : l’en-tête de `tcfds.py` affiche `Fixes vs v6.4.0` puis `Bump to v6.4.0` dans le même bloc, ce qui est auto-référent et ambigu.
- **Tâche proposée** : reformuler le bloc de release notes (ex. `Fixes in v6.4.0` ou `Fixes vs v6.3.x`) et harmoniser toutes les mentions de version.
- **Critères d’acceptation** :
  - libellé non ambigu entre version source et version cible ;
  - cohérence entre docstring, bannière CLI et métadonnées de version.

## 2) Correction de bug (chargement checkpoint compressé)
- **Constat** : la note de release annonce un fix `torch.load(..., weights_only=False)`, mais `load_compressed()` appelle actuellement `torch.load(path, map_location='cpu')` sans cet argument.
- **Risque** : comportement variable selon versions de PyTorch / format de checkpoint (chargement partiel, erreur ou avertissement bloquant).
- **Tâche proposée** : implémenter `weights_only=False` avec fallback compatible si la signature ne supporte pas l’argument, puis remonter un message d’erreur explicite en cas d’échec.
- **Critères d’acceptation** :
  - chargement robuste sur au moins 2 versions de PyTorch supportées ;
  - message d’erreur actionnable si checkpoint invalide/incompatible ;
  - non-régression sur un checkpoint `.pt` généré par `save_compressed()`.

## 3) Correction commentaire / anomalie documentation
- **Constat** : le README indique `tcfds.py  # Main compression script (v6.3)` alors que le projet est en `v6.4.0`.
- **Tâche proposée** : mettre à jour la section *Project Structure* et faire une passe de cohérence versionnelle rapide sur la documentation.
- **Critères d’acceptation** :
  - suppression des mentions obsolètes `v6.3` dans le README ;
  - documentation alignée avec la version courante du package.

## 4) Amélioration des tests (pytest minimal + régressions)
- **Constat** : aucune suite de tests automatisés n’est présente.
- **Tâche proposée** : ajouter une base `pytest` couvrant en priorité :
  - `detect_chat_format`, `format_prompt_for_chat`, `format_ref_for_ppl` (qwen, llama3, phi, mistral, tinyllama),
  - un test de régression `load_compressed()` avec checkpoint mocké,
  - un smoke test du parsing CLI (`argparse`) ;
  - un cas prioritaire sur la divergence de format `phi`.
- **Critères d’acceptation** :
  - `pytest` vert en local ;
  - cas de test nommés et lisibles ;
  - couverture des chemins critiques de formatting + chargement.
