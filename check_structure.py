import os
import torchvision

def check_dataset_structure():
    base_path = "./content/sorted_movie_posters_paligema"
    
    if not os.path.exists(base_path):
        print("❌ ERREUR: Dossier 'sorted_movie_posters_paligema' non trouvé!")
        print("Place-le dans le même dossier que ce script")
        return False
    
    print("✅ Dossier trouvé:", os.path.abspath(base_path))
    
    # Lister tous les dossiers (genres)
    items = os.listdir(base_path)
    genres = [d for d in items if os.path.isdir(os.path.join(base_path, d))]
    
    if not genres:
        print("❌ Aucun dossier (genre) trouvé!")
        return False
    
    print(f"\n📁 {len(genres)} genres trouvés:")
    
    total_images = 0
    for genre in sorted(genres):
        genre_path = os.path.join(base_path, genre)
        images = [f for f in os.listdir(genre_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        
        print(f"  • {genre}: {len(images)} images")
        total_images += len(images)
        
        # Afficher quelques exemples
        if images:
            print(f"    Exemples: {', '.join(images[:2])}")
        else:
            print(f"    ⚠️  Aucune image!")
    
    print(f"\n📊 Total: {total_images} images")
    
    # Test avec ImageFolder
    try:
        dataset = torchvision.datasets.ImageFolder(root=base_path)
        print(f"\n Test ImageFolder réussi!")
        print(f"   Classes: {dataset.classes}")
        print(f"   Class mapping: {dataset.class_to_idx}")
        print(f"   Nombre d'images: {len(dataset)}")
        return True
    except Exception as e:
        print(f"\n❌ Erreur avec ImageFolder: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Vérification de la structure du dataset...")
    success = check_dataset_structure()
    
    if success:
        print("\n🎯 Tout est prêt! Tu peux maintenant:")
        print("1. Entraîner le modèle: python train.py")
        print("2. Lancer l'API: python movie_api.py")
        print("3. Lancer l'interface: python movie_gradio.py")
        print("4. Ou tout lancer avec: docker-compose up")
    else:
        print("\n Problèmes détectés. Corrige-les avant de continuer.")