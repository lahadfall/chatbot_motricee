import streamlit as st
import nltk
import speech_recognition as sr
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from nltk.chat.util import Chat, reflections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
import numpy as np
import pyttsx3 as ttx #pour faire parler l'assistant
import datetime
import webbrowser
import threading
import os

# Liste des ressources à télécharger
nltk_resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']

# Vérification et téléchargement si nécessaire
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')  # Exemple pour les tokenizers
    except LookupError:
        nltk.download(resource)

#nltk.data.path.append(r'nltk_data')
# Télécharger les ressources nécessaires
#nltk.download('punkt')
#nltk.download('stopwords')                                   
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

# Exemple de données textuelles et de leurs catégories
data = [
    # Support technique
    ("Mon produit est défectueux, je veux un remboursement", "support technique"),
    ("Le produit ne correspond pas à la description", "support technique"),
    ("Le produit ne fonctionne pas", "support technique"),
    ("Mon appareil est cassé, comment puis-je le réparer?", "support technique"),
    ("Je n'arrive pas à connecter mon appareil au Wi-Fi", "support technique"),
    ("L'écran est brisé, puis-je l'échanger?", "support technique"),
    ("Mon application ne fonctionne pas correctement", "support technique"),
    ("J'ai un problème avec le logiciel installé", "support technique"),
    ("Le bouton de mon appareil ne répond pas", "support technique"),
    ("Mon produit surchauffe, que faire?", "support technique"),
    ("Je veux signaler un défaut sur mon appareil", "support technique"),
    ("Mon appareil fait un bruit étrange", "support technique"),
    ("Je ne peux pas installer la mise à jour", "support technique"),
    ("Mon produit est en panne, que faire?", "support technique"),
    ("Je veux retourner mon produit pour réparation", "support technique"),
    ("Je n'arrive pas à allumer mon appareil", "support technique"),
    ("Le chargement ne fonctionne pas correctement", "support technique"),
    ("Mon appareil se bloque souvent", "support technique"),
    ("Le son ne fonctionne pas sur mon appareil", "support technique"),
    ("Je reçois des erreurs lors de l'utilisation", "support technique"),
    ("L'application se ferme automatiquement", "support technique"),
    ("Mon produit s'éteint tout seul", "support technique"),
    ("Le produit est très lent", "support technique"),
    ("L'écran tactile ne fonctionne plus", "support technique"),
    ("Mon appareil ne se connecte pas à Internet", "support technique"),
    ("Je ne trouve pas une fonctionnalité dans le produit", "support technique"),
    ("Je n'ai pas reçu de manuel d'instructions", "support technique"),
    ("Mon produit ne s'allume plus après une mise à jour", "support technique"),
    ("Je n'arrive pas à synchroniser mon appareil", "support technique"),
    ("Mon appareil est tombé en panne après 2 mois", "support technique"),
    
    # Expédition
    ("Quand vais-je recevoir ma commande?", "expédition"),
    ("Quand est-ce que ma commande sera expédiée?", "expédition"),
    ("Ma commande n'est toujours pas arrivée", "expédition"),
    ("Combien de temps avant que ma commande soit livrée?", "expédition"),
    ("Je n'ai pas reçu mon numéro de suivi", "expédition"),
    ("Le suivi indique que ma commande est bloquée", "expédition"),
    ("Où se trouve mon colis?", "expédition"),
    ("Mon colis a été livré à la mauvaise adresse", "expédition"),
    ("Puis-je modifier l'adresse de livraison?", "expédition"),
    ("Est-ce que je peux programmer une nouvelle date de livraison?", "expédition"),
    ("Mon colis est endommagé, que faire?", "expédition"),
    ("Je veux annuler la commande avant expédition", "expédition"),
    ("Mon colis est marqué comme livré, mais je ne l'ai pas reçu", "expédition"),
    ("Ma commande est en retard", "expédition"),
    ("Le transporteur ne trouve pas mon adresse", "expédition"),
    ("Comment puis-je suivre mon colis?", "expédition"),
    ("Je veux changer le mode de livraison", "expédition"),
    ("Pourquoi ma commande prend-elle autant de temps?", "expédition"),
    ("Le suivi n'a pas été mis à jour", "expédition"),
    ("J'ai reçu un avis de livraison manquée, que faire?", "expédition"),
    ("Puis-je récupérer ma commande directement au centre de livraison?", "expédition"),
    ("La livraison express est-elle disponible?", "expédition"),
    ("Je veux passer à une livraison plus rapide", "expédition"),
    ("Combien de jours de livraison sont nécessaires?", "expédition"),
    ("Ma commande a été renvoyée à l'expéditeur", "expédition"),
    ("Comment retourner un produit après livraison?", "expédition"),
    ("Est-ce que je dois être présent pour la livraison?", "expédition"),
    ("Puis-je envoyer ma commande à une autre personne?", "expédition"),
    ("Le statut de ma commande n'a pas changé depuis plusieurs jours", "expédition"),
    ("Puis-je choisir un créneau horaire pour la livraison?", "expédition"),
    
    # Facturation
    ("J'ai une question concernant ma facture", "facturation"),
    ("Comment puis-je payer ma facture?", "facturation"),
    ("Je n'arrive pas à accéder à ma facture en ligne", "facturation"),
    ("Pourquoi ma facture est-elle plus élevée que d'habitude?", "facturation"),
    ("Je ne comprends pas certains frais sur ma facture", "facturation"),
    ("Je n'ai pas reçu ma facture ce mois-ci", "facturation"),
    ("Comment puis-je contester des frais sur ma facture?", "facturation"),
    ("Puis-je obtenir une copie de ma facture?", "facturation"),
    ("Y a-t-il des options de paiement automatique?", "facturation"),
    ("Quels moyens de paiement acceptez-vous?", "facturation"),
    ("Pourquoi ai-je reçu une facture après annulation?", "facturation"),
    ("Je souhaite modifier les informations de facturation", "facturation"),
    ("Ma facture indique un solde incorrect", "facturation"),
    ("Je veux payer ma facture en plusieurs fois", "facturation"),
    ("Je n'ai pas reçu de confirmation après mon paiement", "facturation"),
    ("Pouvez-vous me renvoyer ma facture?", "facturation"),
    ("Je veux recevoir mes factures par e-mail", "facturation"),
    ("Puis-je modifier la date d'échéance de ma facture?", "facturation"),
    ("J'ai été facturé deux fois pour la même commande", "facturation"),
    ("Puis-je obtenir un remboursement pour des frais non autorisés?", "facturation"),
    ("Comment mettre à jour ma carte de crédit pour la facturation?", "facturation"),
    ("Ma facture affiche des frais supplémentaires que je ne comprends pas", "facturation"),
    ("J'ai été facturé pour un produit que je n'ai pas commandé", "facturation"),
    ("Je souhaite désactiver la facturation automatique", "facturation"),
    ("Mon compte a été débité mais je ne vois pas ma facture payée", "facturation"),
    ("Pouvez-vous prolonger le délai de paiement?", "facturation"),
    ("Est-il possible de recevoir un rappel avant l'échéance?", "facturation"),
    ("Comment puis-je vérifier le statut de ma facture?", "facturation"),
    ("Je ne comprends pas les détails des taxes sur ma facture", "facturation"),
    ("Pourquoi ma facture n'a-t-elle pas été mise à jour après le paiement?", "facturation")
]

# Initialisation des outils
stop_words = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Fonction pour mapper les étiquettes POS vers celles de WordNet
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Par défaut, on retourne "nom" si l'étiquette POS n'est pas reconnue

def preprocess(text):
    # Mise en minuscule et tokenisation
    tokens = word_tokenize(text.lower())
    
    # Balisage (POS tagging)
    pos_tags = pos_tag(tokens)
    
    # Lemmatisation et suppression des stopwords
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
        for word, tag in pos_tags 
        if word.isalnum() and word not in stop_words
    ]
    
    # Application de la dérivation (stemming)
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
    
    # Retourne la phrase prétraitée
    return ' '.join(stemmed_words)

# Appliquer le prétraitement aux données
texts, labels = zip(*data)
texts = [preprocess(text) for text in texts]

# Vectorisation des textes
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Fonction pour classifier un texte
def classify_text(model, text):
    preprocessed_text = preprocess(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    return model.predict(vectorized_text)[0]

# Fonction pour gérer la reconnaissance vocale
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Parlez maintenant...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language="fr-FR")
        return text
    except sr.UnknownValueError:
        return "Je n'ai pas compris l'audio"
    except sr.RequestError as e:
        return f"Erreur du service de reconnaissance vocale: {e}"

# Sidebar pour ajuster les hyperparamètres
st.sidebar.title("Paramètres Généreaux  ")
# Sidebar pour la navigation entre pages

page = st.sidebar.selectbox("Navigation :", ["Classification", "Chatbot Motrice"])

# Option de choix du modèle de classification
model_option = st.sidebar.selectbox("Choisissez le modèle de classification :", ["Arbre de décision", "Bayésien naïf"])

# Hyperparamètres
# Exemple d'utilisation d'icône pour les hyperparamètres
#st.sidebar.markdown('<i class="fa fa-sliders" style="font-size:24px;color:blue;"></i> Paramètres du modèle', unsafe_allow_html=True)

if model_option == "Arbre de décision":
    entropy_threshold = st.sidebar.slider("Seuil d'entropie (min_samples_split)", 2, 10, 5)
elif model_option == "Bayésien naïf":
    alpha_nb = st.sidebar.slider("Alpha pour Bayésien naïf", 0.1, 1.0, 0.5, step=0.1)
else:
    st.sidebar.write("choisir entre 'Arbre de décision' et 'Bayésien naïf'")
# Option pour choisir la métrique
metric_option = st.sidebar.selectbox("Choisissez une métrique :", ["MSE", "Roc_curve", "Matrice de confusion"])


# Ajustement dynamique des modèles
if model_option == "Arbre de décision":
    classifier = DecisionTreeClassifier(criterion='entropy', min_samples_split=entropy_threshold)
elif model_option == "Bayésien naïf":
    classifier = MultinomialNB(alpha=alpha_nb)

# Entraîner le modèle avec les hyperparamètres choisis
classifier.fit(X_train, y_train)

# Affichage de la matrice de confusion ou MSE
if metric_option == "MSE":
    y_pred = classifier.predict(X_test)
    mse = mean_squared_error([1 if y == "support technique" else 0 for y in y_test], 
                                 [1 if y == "support technique" else 0 for y in y_pred])
    #st.sidebar.write(f"MSE : **{mse:.2f}**")

if page == "Classification":

    # Interface principale
    st.title("Chatbot de Classification de Texte")
    # Ajout d'une image dans la section principale
    st.image("chat2_img-remov.png", caption="",width=400)
    
    st.write("## Saisir un texte ou utiliser une commande vocale")
    
    # Option de saisie de texte ou vocale
    option = st.selectbox("Choisissez la méthode de saisie :", ["Texte", "Vocale"])
    user_input = " "
    if option == "Texte":
        user_input = st.text_input("Entrez votre texte ici :")
        if st.button("Classer le texte"):  # Utiliser un seul bouton pour la classification
            if user_input:
                category = classify_text(classifier, user_input)
                st.success(f"Le texte appartient à la catégorie : **{category}**")
                st.balloons()
            else:
                st.write("Veuillez entrer un texte ou utiliser la commande vocale.")
    elif option == "Vocale":
        if st.button("Utiliser le microphone"):
            user_input = recognize_speech()
            st.write(f"Texte reconnu : {user_input}")
            # Classer immédiatement après la reconnaissance vocale
            if user_input:
                category = classify_text(classifier, user_input)
                st.success(f"Le texte appartient à la catégorie : **{category}**")

        
    #  Afficher la précision du modèle                               
    if st.button("Afficher la précision"):
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Précision du modèle {model_option} : **{accuracy:.2f}**")
      
    
elif page == "Chatbot Motrice":
    # st.title("Autre Application ")

    engine = ttx.init()
    voice = engine.getProperty("voices")
    engine.setProperty('voice', 'french')
    # engine.setProperty('voice', voice[1].id) si c'est en anglais
    
    
    def parler(text):
        engine.endLoop()  # Fin de toute boucle en cours
        engine.say(text)
        engine.runAndWait()   


    # Fonction de prétraitement
    def preprocess(text):
        tokens = nltk.word_tokenize(text)
        return ' '.join(tokens)

    # Reconnaissance vocale
    def recognize_speech():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Parlez maintenant...")
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="fr-FR")
            return text
        except sr.UnknownValueError:
            return "Désolé, je n'ai pas compris."
        except sr.RequestError:
            return "Le service de reconnaissance vocale n'est pas disponible pour le moment."

    # Ouvrir YouTube
    def open_youtube():
        webbrowser.open("https://www.youtube.com")
        return "Ouverture de YouTube..."
    
    # Ouvrir une vidéo 
    def open_youtube_video(auteur):
        query = auteur.replace(' ', '+')  # Formatage de la requête
        youtube_search_url = f"https://www.youtube.com/results?search_query={query}"
        webbrowser.open(youtube_search_url)

    # message WhatsApp 
    def send_whatsapp_message(phone_number, message):
        whatsapp_url = f"https://web.whatsapp.com/send?phone={phone_number}&text={message}"
        webbrowser.open(whatsapp_url)
    
    # Chercher sur YouTube
    def search_youtube(query):
        search_query = "+".join(query.split())
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        return f"Recherche de '{query}' sur YouTube..."

    # Donner l'heure actuelle
    def current_time():
        now = datetime.datetime.now()
        return now.strftime("%H:%M:%S")

    # Chercher sur Google
    def search_google(query):
        search_query = "+".join(query.split())
        webbrowser.open(f"https://www.google.com/search?q={search_query}")
        return f"Recherche de '{query}' sur Google..."
    

    # Fonction du chatbot
    def chatbot_response(user_input):
        if "bonjour " in user_input.lower():
            parler("Bonjour ça va, Que voulez-vous? ")
        elif "salut " in user_input.lower():
            parler("salut toi! comment pourrais-je vous aider ")
        elif "heure" in user_input.lower():
            return f"L'heure actuelle est {current_time()}."
        elif "fatigué" in user_input.lower():
            heure = datetime.datetime.now().strftime("%H:%M")
            parler('désolé  il fait'+heure)
        elif "cherche" in user_input.lower():
            query = user_input.lower().replace("cherche", "").strip()
            return search_google(query) 
        elif "youtube" in user_input.lower() and "cherche" in user_input.lower():
            query = user_input.lower().replace("cherche", "").replace("sur youtube", "").strip()
            return search_youtube(query)
        elif 'mets la vidéo de' in user_input.lower():
            auteur = user_input.lower().replace('mets la vidéo de', '').strip()
            st.write(auteur)
            open_youtube_video(auteur)
        elif "youtube" in user_input.lower():
            return open_youtube()
        elif "rappeler" in user_input.lower():
            send_whatsapp_message("+221766342536", "Bonjour!")  # Envoie un message à 15h30
        elif "merci" in user_input.lower():
            parler("merci passez une bonne journée, à la prochaine")
        else:
            return "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ?"

    # Application principale
    def main():
        
        st.title("ChatBot Motrice")
        st.write("## Ceci est un autre ChatBot permettant d'accès rapide aux moteurs de recherches.")
        # Ajout d'une image dans la section principale
        
        # Barre latérale
        st.sidebar.title("Options")
        st.sidebar.write("Utilisez la barre latérale pour les paramètres supplémentaires.")

        # Affichage dans la barre latérale
        st.sidebar.write("Vous pouvez parler ou taper des commandes pour interagir avec le chatbot.")
        st.sidebar.write("Commandes disponibles :")
        st.sidebar.write("- 'Bonjour/Salut'")
        st.sidebar.write("- 'donne l'heure'")
        st.sidebar.write("- 'ouvre YouTube'")
        st.sidebar.write("- 'cherche [terme] sur YouTube'")
        st.sidebar.write("- 'cherche [terme] sur Google'")
        st.sidebar.write("- 'Merci'")

        # Saisie textuelle ou vocale
        user_input = st.text_input("**Entrez votre texte ici ou parlez en appuyant sur le bouton 'Micro'**.")
        
        if st.button("Micro"):
            user_input = recognize_speech()
            st.write(f"Vous avez dit : {user_input}")

        if user_input:
            response = chatbot_response(user_input)
            st.write(response)

        
        st.image("chat_img-remove.png", caption="",width=300)
        
    if __name__ == "__main__":
        main()



