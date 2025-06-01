import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq
import os
import re
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt

"""
Streamlit ESG Investor Profiling App
------------------------------------
Esta versión garantiza que **todas las preguntas abiertas** (preguntas iniciales y
reacciones a noticias) lancen **exactamente una** pregunta de seguimiento si la
primera respuesta es considerada "pobre".  Para las preguntas iniciales, además
se aplica un umbral mínimo de longitud para evitar que un "True" demasiado
permisivo impida el seguimiento.
"""

# ---------------------------------------------------------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------------------------------------------------------

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

preguntas_inversor = [
    "¿Cuál es tu objetivo principal al invertir?",
    "¿Cuál es tu horizonte temporal de inversión?",
    "¿Tienes experiencia previa invirtiendo en activos de mayor riesgo como acciones, criptomonedas o fondos alternativos?",
    "¿Estás dispuesto a sacrificar parte de la rentabilidad potencial a cambio de un impacto social o ambiental positivo?",
    "¿Qué opinas sobre el cambio climático?",
]

noticias = [
    "Repsol, entre las 50 empresas que más responsabilidad histórica tienen en el calentamiento global",
    "Amancio Ortega crea un fondo de 100 millones de euros para los afectados de la dana",
    "Freshly Cosmetics despide a 52 empleados en Reus, el 18% de la plantilla",
    "Wall Street y los mercados globales caen ante la incertidumbre por la guerra comercial y el temor a una recesión",
    "El mercado de criptomonedas se desploma: Bitcoin cae a 80.000 dólares, las altcoins se hunden en medio de una frenética liquidación",
]

# ---------------------------------------------------------------------------
# PLANTILLAS LLM
# ---------------------------------------------------------------------------

plantilla_evaluacion = """
Evalúa si esta respuesta del usuario es suficientemente detallada para un análisis ESG.
Criterios:
- Claridad de la opinión
- Especificidad respecto a la noticia o la pregunta original
- Mención de aspectos ESG (ambiental, social, gobernanza o riesgo)
- Identificación de preocupaciones o riesgos

Respuesta del usuario: {respuesta}

Si es vaga o superficial, responde "False".
Si contiene opinión sustancial y analizable, responde "True".

Solo responde "True" o "False".
"""

prompt_evaluacion = PromptTemplate(template=plantilla_evaluacion, input_variables=["respuesta"])
cadena_evaluacion = LLMChain(llm=llm, prompt=prompt_evaluacion)

plantilla_reaccion = """
Reacción del inversor: {reaccion}
Genera ÚNICAMENTE una pregunta de seguimiento enfocada en profundizar en su opinión.
"""

prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

plantilla_perfil = """
Análisis de respuestas: {analisis}
Genera un perfil detallado del inversor basado en sus respuestas, enfocándote en los pilares ESG (Ambiental, Social y Gobernanza) y su aversión al riesgo.
Asigna una puntuación de 0 a 100 para cada pilar ESG y para el riesgo, donde 0 indica ninguna preocupación y 100 máxima preocupación o aversión.
Devuelve las 4 puntuaciones en formato: Ambiental: [puntuación], Social: [puntuación], Gobernanza: [puntuación], Riesgo: [puntuación]
"""

prompt_perfil = PromptTemplate(template=plantilla_perfil, input_variables=["analisis"])
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------------

def generar_pregunta_seguimiento(respuesta: str) -> str:
    """Devuelve una única pregunta de seguimiento."""
    return cadena_reaccion.run(reaccion=respuesta).strip()


def respuesta_es_pobre(respuesta: str, umbral_longitud: int = 25) -> bool:
    """Combina evaluación LLM + longitud mínima."""
    evaluacion = cadena_evaluacion.run(respuesta=respuesta).strip().lower()
    es_llm_pobre = "false" in evaluacion
    es_corta = len(respuesta.strip()) < umbral_longitud
    return es_llm_pobre or es_corta

# ---------------------------------------------------------------------------
# ESTADO DE SESIÓN
# ---------------------------------------------------------------------------

if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.reacciones = []
    st.session_state.pregunta_general_idx = 0
    st.session_state.noticia_idx = 0
    st.session_state.pregunta_pendiente = False
    st.session_state.perfil_valores = {}
    st.session_state.cuestionario_enviado = False

# ---------------------------------------------------------------------------
# INTERFAZ DE CHAT
# ---------------------------------------------------------------------------

st.title("Chatbot de Análisis de Inversor ESG")

st.markdown(
    """
    **Primero interactuarás con un chatbot para evaluar tu perfil ESG.**  
    **Al final, completarás un test tradicional de perfilado.**
    """
)

# Mostrar historial completo
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"], avatar="🤖" if mensaje["tipo"] == "bot" else None):
        st.write(mensaje["contenido"])

# ---------------------------------------------------------------------------
# BLOQUE 1: PREGUNTAS INICIALES
# ---------------------------------------------------------------------------

if st.session_state.pregunta_general_idx < len(preguntas_inversor):
    pregunta_actual = preguntas_inversor[st.session_state.pregunta_general_idx]

    # Mostrar la pregunta si aún no está en el historial
    if not any(
        p["contenido"] == pregunta_actual for p in st.session_state.historial if p["tipo"] == "bot"
    ):
        st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_actual})
        with st.chat_message("bot", avatar="🤖"):
            st.write(pregunta_actual)

    # Capturar respuesta del usuario
    user_input = st.chat_input("Escribe tu respuesta aquí...")

    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})

        # -------- Respuesta a una pregunta de seguimiento pendiente --------
        if st.session_state.pregunta_pendiente:
            st.session_state.reacciones.append(user_input)
            st.session_state.pregunta_general_idx += 1
            st.session_state.pregunta_pendiente = False
            st.rerun()

        # -------- Primera respuesta: evaluar suficiencia --------
        if respuesta_es_pobre(user_input):
            pregunta_seguimiento = generar_pregunta_seguimiento(user_input)
            st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_seguimiento})
            with st.chat_message("bot", avatar="🤖"):
                st.write(pregunta_seguimiento)
            st.session_state.pregunta_pendiente = True  # Esperar esta respuesta
        else:
            st.session_state.reacciones.append(user_input)
            st.session_state.pregunta_general_idx += 1
            st.rerun()

# ---------------------------------------------------------------------------
# BLOQUE 2: NOTICIAS ESG
# ---------------------------------------------------------------------------

elif st.session_state.noticia_idx < len(noticias):
    noticia_actual = noticias[st.session_state.noticia_idx]

    if not any(
        p["contenido"].startswith("¿Qué opinas sobre esta noticia?") and noticia_actual in p["contenido"]
        for p in st.session_state.historial
        if p["tipo"] == "bot"
    ):
        texto_noticia = f"¿Qué opinas sobre esta noticia? {noticia_actual}"
        st.session_state.historial.append({"tipo": "bot", "contenido": texto_noticia})
        with st.chat_message("bot", avatar="🤖"):
            st.write(texto_noticia)

    user_input = st.chat_input("Escribe tu respuesta aquí...")

    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})

        # Respuesta al seguimiento de la noticia
        if st.session_state.pregunta_pendiente:
            st.session_state.reacciones.append(user_input)
            st.session_state.noticia_idx += 1
            st.session_state.pregunta_pendiente = False
            st.rerun()

        # Evaluar primera reacción a la noticia
        if respuesta_es_pobre(user_input):
            pregunta_seguimiento = generar_pregunta_seguimiento(user_input)
            st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_seguimiento})
            with st.chat_message("bot", avatar="🤖"):
                st.write(pregunta_seguimiento)
            st.session_state.pregunta_pendiente = True
        else:
            st.session_state.reacciones.append(user_input)
            st.session_state.noticia_idx += 1
            st.rerun()

# ---------------------------------------------------------------------------
# BLOQUE 3: PERFIL FINAL Y CUESTIONARIO
# ---------------------------------------------------------------------------

else:
    # Perfil ESG (solo se calcula una vez)
    if not st.session_state.perfil_valores:
        analisis_total = "\n".join(st.session_state.reacciones)
        perfil = cadena_perfil.run(analisis=analisis_total)

        st.session_state.perfil_valores = {
            "Ambiental": int(re.search(r"Ambiental: (\d+)", perfil).group(1)),
            "Social": int(re.search(r"Social: (\d+)", perfil).group(1)),
            "Gobernanza": int(re.search(r"Gobernanza: (\d+)", perfil).group(1)),
            "Riesgo": int(re.search(r"Riesgo: (\d+)", perfil).group(1)),
        }

    # Mostrar perfil
    with st.chat_message("bot", avatar="🤖"):
        st.write(
            f"**Perfil del inversor:** "
            f"Ambiental: {st.session_state.perfil_valores['Ambiental']}, "
            f"Social: {st.session_state.perfil_valores['Social']}, "
            f"Gobernanza: {st.session_state.perfil_valores['Gobernanza']}, "
            f"Riesgo: {st.session_state.perfil_valores['Riesgo']}"
        )

    # Gráfico de barras
    fig, ax = plt.subplots()
    ax.bar(st.session_state.perfil_valores.keys(), st.session_state.perfil_valores.values())
    ax.set_ylabel("Puntuación (0-100)")
    ax.set_title("Perfil del Inversor")
    st.pyplot(fig)

    # -------- FORMULARIO FINAL --------
    if not st.session_state.cuestionario_enviado:
        st.header("Cuestionario Final de Perfilado")

        with st.form("formulario_final"):
            objetivo = st.radio(
                "2.1. ¿Cuál es tu objetivo principal al invertir?",
                [
                    "Preservar el capital (bajo riesgo)",
                    "Obtener rentabilidad moderada",
                    "Maximizar la rentabilidad (alto riesgo)",
                ],
                index=None,
            )
            horizonte = st.radio(
                "2.2. ¿Cuál es tu horizonte temporal de inversión?",
                [
                    "Menos de 1 año",
                    "Entre 1 y 5 años",
                    "Más de 5 años",
                ],
                index=None,
            )
            productos = st.multiselect(
                "3.1. ¿Qué productos financieros conoces o has utilizado?",
                [
                    "Cuentas de ahorro",
                    "Fondos de inversión",
                    "Acciones",
                    "Bonos",
                    "Derivados (futuros, opciones, CFD)",
                    "Criptomonedas",
                ],
            )
            volatil
