import os
import sys
import time
import data_collector
import import_liquipedia

def main():
    filename = "players.txt"
    
    if not os.path.exists(filename):
        print(f"Erreur : Le fichier '{filename}' n'existe pas.")
        print(f"Veuillez créer un fichier '{filename}' avec un joueur par ligne (ex: Faker#KR1)")
        sys.exit(1)

    player_list = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Ignorer les lignes vides ou les commentaires
            if not line or line.startswith("//") or line.startswith(";"):
                continue
            
            if "#" in line:
                name, tag = line.split("#", 1)
                player_list.append((name.strip(), tag.strip()))
            else:
                print(f"[*] Resolving pro player '{line}' via Liquipedia...")
                resolved = import_liquipedia.resolve_pro_name(line)
                if resolved:
                    player_list.append(resolved)
                else:
                    print(f"[!] Could not resolve '{line}'")
                time.sleep(2)

    if not player_list:
        print(f"Aucun joueur valide trouvé dans {filename}.")
        sys.exit(1)

    print(f"============================================================")
    print(f"  Lancement de la collecte batch pour {len(player_list)} joueurs")
    print(f"  Recherche de smurfs sur : Liquipedia, Lolpros")
    print(f"  Objectif : 50 matchs par compte (si > 5 matchs joués)")
    print(f"============================================================\n")

    sources = ["liquipedia", "lolpros"]
    matches_per_account = 50

    # L'exécution est synchrone. Les print() de data_collector.add_batch_log
    # s'afficheront directement dans la console en temps réel.
    try:
        data_collector.collect_batch_with_smurfs(player_list, sources, count=matches_per_account)
    except KeyboardInterrupt:
        print("\n\n[!] Collecte annulée par l'utilisateur.")
        sys.exit(0)

if __name__ == "__main__":
    main()
