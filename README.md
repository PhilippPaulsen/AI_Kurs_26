# 🤖 AI Agent Playground (Russell & Norvig Edition)

Willkommen im interaktiven Kurs-Tool für den Kurs **AI 2026**. Diese Web-App wurde entwickelt, um die theoretischen Konzepte von intelligenten Agenten und Reinforcement Learning (RL) nach dem Standardwerk von Russell & Norvig greifbar zu machen – ganz ohne Programmierkenntnisse.

---

## 🎯 Lernziele
* **Agenten-Typen:** Den Unterschied zwischen Reflex-, Model-based- und Goal-based-Agents verstehen.
* **RL-Grundlagen:** Die Auswirkung von Parametern wie Lernrate ($\alpha$) und Exploration ($\epsilon$) live beobachten.
* **MDPs:** Das Konzept der Markov-Entscheidungsprozesse in einem Grid-World-Szenario erleben.

## 🛠 Installation & Start
Um die App lokal zu starten, folgen Sie diesen Schritten im Terminal:

1. **Abhängigkeiten installieren:**
   ```bash
   pip install -r requirements.txt

2. **App ausführen:**
   ```bash
   streamlit run app.py

   https://ai-kurs26-agenten.streamlit.app

   🎮 Das Spielfeld (ASCII Grid)
Die Welt besteht aus einem dynamischen Gitter, das in der App angepasst werden kann:

A = Agent (Der Algorithmus in Aktion)

G = Goal (Das Ziel / Positive Belohnung)

# = Wall (Hindernis / Undurchlässig)

. = Path (Begehbares Feld)

🧠 Implementierte Agenten-Modelle
1. Simple Reflex Agent

Arbeitet nach dem "Condition-Action"-Prinzip. Er sieht nur seine unmittelbare Nachbarschaft und handelt nach festen Regeln (z.B. "Wenn Weg frei, gehe vor").

Schwachstelle: Kann in Sackgassen oder unendlichen Schleifen hängen bleiben, da er kein Gedächtnis hat.

2. Model-based Reflex Agent

Besitzt einen internen Zustand (Memory). Er merkt sich, welche Felder er bereits besucht hat oder wie die Welt aussieht, um Sackgassen effizienter zu vermeiden.

3. Learning Agent (Q-Learning)

Dieser Agent kennt die Regeln der Welt anfangs nicht. Er lernt durch Belohnung und Bestrafung (Trial and Error).

Die Logik dahinter (Bellman-Gleichung):

Q(s,a)←Q(s,a)+α[r+γ 
a 
′
 
max
​	
 Q(s 
′
 ,a 
′
 )−Q(s,a)]
Visualisierung: Die "Erfahrung" des Agenten wird unter dem Spielfeld als Heatmap der Q-Werte (Nutzenwerte pro Feld) dargestellt.

🤖 Instruktion für Antigravity / AI Studio:

Bitte erweitere diese Dokumentation basierend auf dem generierten Code:

Erkläre die spezifischen UI-Elemente (Slider, Buttons).

Beschreibe, wie das "Log-Fenster" die Denkprozesse des Agenten für Studierende übersetzt.

Füge eine kurze Anleitung hinzu, wie Nutzer eigene Grid-Layouts direkt in der App definieren können.