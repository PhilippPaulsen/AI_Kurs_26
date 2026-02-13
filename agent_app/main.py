import streamlit as st
import sys

# DEBUG: Zeige sofort etwas an, bevor die Logik startet
st.title("Debug Mode")
st.write("Python Version:", sys.version)

try:
    import pandas as pd
    import numpy as np
    import random
    
    st.success("Imports erfolgreich!")

    # DEINE LOGIK MINIMAL
    if 'count' not in st.session_state:
        st.session_state.count = 0
    
    if st.button("Klick mich"):
        st.session_state.count += 1
    
    st.write(f"Counter: {st.session_state.count}")
    
    # Hier der Rest deines Codes...
    class Grid:
        def __init__(self, w, h):
            self.w, self.h = w, h
            self.grid = np.zeros((h, w))
    
    g = Grid(5, 5)
    st.write("Grid Objekt erstellt.")

except Exception as e:
    st.error(f"Kritischer Fehler beim Starten: {e}")
    st.exception(e)