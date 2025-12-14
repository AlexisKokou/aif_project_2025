#!/bin/bash

echo "🔧 Réparation du déploiement Docker"
echo "======================================"

echo "1. Arrêt des services..."
docker-compose down

echo "2. Nettoyage Docker..."
docker system prune -f

echo "3. Reconstruction des images..."
docker-compose build

echo "4. Démarrage des services..."
docker-compose up -d

echo "5. Attente du démarrage..."
sleep 10

echo "6. Vérification des services..."
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services démarrés"
else
    echo "❌ Certains services ne sont pas démarrés"
    docker-compose ps
fi

echo "7. Test de l'API..."
if curl -s http://localhost:5075/health > /dev/null; then
    echo "✅ API fonctionnelle"
    curl -s http://localhost:5075/health | python -m json.tool
else
    echo "❌ API non fonctionnelle"
    echo "Logs API :"
    docker-compose logs api --tail=20
fi

echo "8. Test de Gradio..."
if curl -s http://localhost:7860 > /dev/null; then
    echo "✅ Interface Gradio accessible"
else
    echo "⚠️  Gradio peut être en cours de démarrage"
fi

echo "======================================"
echo "Pour suivre les logs :"
echo "  docker-compose logs -f"
echo "Pour arrêter :"
echo "  docker-compose down"