import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import qrcode
from io import BytesIO
import sqlite3
from datetime import datetime

st.set_page_config(page_title="INTEGRaL 360 - Dashboard Dueño", layout="wide")
st.title("☕ INTEGRaL 360: Dashboard para Café Integral")
st.markdown("Analiza tu negocio, mercado y predice el futuro. ¡Datos vivos y accionables!")

# Conexión DB
@st.cache_resource
def load_db():
    try:
        conn = sqlite3.connect('integral360.db')
        ventas = pd.read_sql('SELECT * FROM ventas', conn)
        costos = pd.read_sql('SELECT * FROM costos', conn)
        clientes = pd.read_sql('SELECT * FROM clientes', conn)
        socio = pd.read_sql('SELECT * FROM sociodemografica', conn)
        pred = pd.read_sql('SELECT * FROM predicciones', conn)
        conn.close()
        
        # Validar tabla ventas
        if ventas.empty:
            st.error("Error: La tabla 'ventas' está vacía. Regenera la base de datos con init_db.py.")
            st.stop()
        expected_columns = ['id', 'fecha', 'producto', 'cantidad', 'precio_unitario', 'total', 'cliente_id', 'hora', 'costo_insumo']
        if not all(col in ventas.columns for col in expected_columns):
            st.error(f"Error: La tabla 'ventas' no contiene todas las columnas esperadas. Encontradas: {list(ventas.columns)}. Regenera la base de datos con init_db.py.")
            st.stop()
        # Validar cliente_id
        try:
            ventas['cliente_id'] = pd.to_numeric(ventas['cliente_id'], errors='coerce').astype('Int64')
            if ventas['cliente_id'].isna().any():
                st.error(f"Error: {ventas['cliente_id'].isna().sum()} valores nulos en 'cliente_id'. Regenera la base de datos.")
                st.stop()
            if not ventas['cliente_id'].between(1, 100).all():
                st.error(f"Error: Algunos valores en 'cliente_id' están fuera del rango 1-100. Regenera la base de datos.")
                st.stop()
        except Exception as e:
            st.error(f"Error al validar 'cliente_id' en ventas: {str(e)}")
            st.stop()
        # Validar fecha
        try:
            ventas['fecha'] = pd.to_datetime(ventas['fecha'], errors='coerce').dt.strftime('%Y-%m-%d')
            if ventas['fecha'].isna().any():
                st.error(f"Error: {ventas['fecha'].isna().sum()} valores nulos en 'fecha'. Regenera la base de datos.")
                st.stop()
        except Exception as e:
            st.error(f"Error al validar 'fecha' en ventas: {str(e)}")
            st.stop()
        # Asegurar que clientes['id'] sea int
        try:
            clientes['id'] = pd.to_numeric(clientes['id'], errors='coerce').astype('Int64')
            if clientes['id'].isna().any():
                st.error(f"Error: {clientes['id'].isna().sum()} valores nulos en clientes['id']. Regenera la base de datos.")
                st.stop()
        except Exception as e:
            st.error(f"Error al validar clientes['id']: {str(e)}")
            st.stop()
        return ventas, costos, clientes, socio, pred
    except Exception as e:
        st.error(f"Error al cargar la base de datos: {str(e)}. Verifica que 'integral360.db' exista en /Users/robertoibarrasuarez/Desktop/demo360.")
        st.stop()

try:
    ventas, costos, clientes, socio, pred = load_db()
except Exception as e:
    st.error(f"Fallo al cargar datos: {str(e)}. Regenera la base de datos con init_db.py.")
    st.stop()

# Sidebar: Carga diaria
with st.sidebar:
    uploaded = st.file_uploader("Cargar ventas diarias (CSV)")
    if uploaded:
        try:
            new_ventas = pd.read_csv(uploaded)
            required_columns = ['fecha', 'producto', 'cantidad', 'precio_unitario', 'total', 'cliente_id', 'hora', 'costo_insumo']
            if not all(col in new_ventas.columns for col in required_columns):
                st.error(f"El CSV debe incluir todas las columnas: {', '.join(required_columns)}")
            else:
                new_ventas['cliente_id'] = pd.to_numeric(new_ventas['cliente_id'], errors='coerce').astype('Int64')
                new_ventas['fecha'] = pd.to_datetime(new_ventas['fecha'], errors='coerce').dt.strftime('%Y-%m-%d')
                if new_ventas['cliente_id'].isna().any():
                    st.error(f"Error: {new_ventas['cliente_id'].isna().sum()} valores nulos en 'cliente_id' del CSV.")
                elif not new_ventas['cliente_id'].between(1, 100).all():
                    st.error("Error: Algunos valores en 'cliente_id' del CSV están fuera del rango 1-100.")
                elif new_ventas['fecha'].isna().any():
                    st.error(f"Error: {new_ventas['fecha'].isna().sum()} valores nulos en 'fecha' del CSV.")
                else:
                    new_ventas.to_sql('ventas', sqlite3.connect('integral360.db'), if_exists='append', index=False)
                    st.success("¡Datos cargados! Recarga la página (presiona 'Rerun').")
        except Exception as e:
            st.error(f"Error al cargar el CSV: {str(e)}. Verifica el formato del archivo.")
    st.button("Generar QR para nuevo cliente")

# Pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Análisis Interno", "🏘️ Sociodemográfico", "🤖 Análisis IA", "🔮 Predictivo", "❤️ Lealtad Clientes"])

# Pestaña 1: Análisis Interno
with tab1:
    st.header("📊 Análisis Interno: Costos, Ventas y Clientes")

    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        default_min = pd.to_datetime(ventas['fecha']).min().date()
        default_max = pd.to_datetime(ventas['fecha']).max().date()
        fecha_min, fecha_max = st.date_input("Rango fecha", [default_min, default_max], min_value=default_min, max_value=default_max)
    with col2:
        producto_filt = st.multiselect("Productos", ventas['producto'].unique())

    # Filtrado
    fecha_min_str = fecha_min.strftime('%Y-%m-%d')
    fecha_max_str = fecha_max.strftime('%Y-%m-%d')
    df_vent_filt = ventas[(ventas['fecha'] >= fecha_min_str) & (ventas['fecha'] <= fecha_max_str)]
    if producto_filt:
        df_vent_filt = df_vent_filt[df_vent_filt['producto'].isin(producto_filt)]

    df_cost_filt = costos[(costos['fecha'] >= fecha_min_str) & (costos['fecha'] <= fecha_max_str)]

    # Validaciones
    if df_vent_filt.empty or df_cost_filt.empty:
        st.warning("⚠️ No hay datos suficientes para el análisis en el rango seleccionado.")
        st.stop()

    # Cálculos base
    total_ventas = df_vent_filt['total'].sum()
    total_costos = df_cost_filt['monto'].sum()
    margen_bruto = ((total_ventas - total_costos) / total_ventas * 100) if total_ventas > 0 else 0
    ventas_diarias = df_vent_filt.groupby('fecha')['total'].sum().mean()
    punto_equilibrio = total_costos / (margen_bruto / 100) if margen_bruto > 0 else 0
    dias_equilibrio = (punto_equilibrio - total_ventas) / ventas_diarias if ventas_diarias > 0 else 0

    # Panel superior de KPIs
    st.subheader("📌 Indicadores Clave")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Margen Bruto", f"{margen_bruto:.1f}%")
    col2.metric("Punto Equilibrio", f"${punto_equilibrio:,.0f}")
    col3.metric("Ticket Promedio", f"${total_ventas / df_vent_filt['cliente_id'].nunique():.0f}")
    col4.metric("Días para Equilibrio", f"{dias_equilibrio:.1f} días")

    # Simulador de meta mensual
    st.subheader("🎯 Simulador de Meta Mensual")
    meta_ingresos = st.slider("Meta de ingresos ($)", 50000, 200000, 100000, step=5000)
    ventas_necesarias = meta_ingresos / (margen_bruto / 100) if margen_bruto > 0 else 0
    st.write(f"Para alcanzar ${meta_ingresos:,.0f}, debe vender aproximadamente ${ventas_necesarias:,.0f} este mes.")

    # Alertas de sobrecosto
    st.subheader("🚨 Alertas de Costo")
    costos_por_tipo = df_cost_filt.groupby('tipo')['monto'].sum().reset_index()
    tipo_mayor = costos_por_tipo.sort_values('monto', ascending=False).iloc[0]
    if tipo_mayor['monto'] > total_costos * 0.4:
        st.warning(f"⚠️ El tipo de costo '{tipo_mayor['tipo']}' representa más del 40% del gasto total. Revise insumos o proveedores.")

    # Comparativo mensual
    st.subheader("📅 Comparativo Mensual")
    ventas['mes'] = pd.to_datetime(ventas['fecha']).dt.to_period('M')
    comp_mensual = ventas.groupby('mes')['total'].sum().reset_index()
    chart_comp = alt.Chart(comp_mensual).mark_bar().encode(
        x='mes:T',
        y='total:Q',
        tooltip=['mes', 'total']
    ).properties(title="Ventas por Mes")
    st.altair_chart(chart_comp, use_container_width=True)

    # Margen por producto
    st.subheader("📦 Margen por Producto")
    df_margen = df_vent_filt.groupby('producto')[['total', 'costo_insumo']].sum()
    df_margen['margen_%'] = ((df_margen['total'] - df_margen['costo_insumo']) / df_margen['total']) * 100
    st.dataframe(df_margen.sort_values('margen_%', ascending=False).reset_index())

    # Horarios de bajo desempeño
    st.subheader("🕒 Horarios de Baja Demanda")
    ventas_hora = df_vent_filt.groupby('hora')['total'].sum().reset_index()
    horas_bajas = ventas_hora[ventas_hora['total'] < ventas_hora['total'].mean()]
    chart_hora = alt.Chart(ventas_hora).mark_line().encode(
        x='hora:N',
        y='total:Q'
    ).properties(title="Ventas por Hora")
    st.altair_chart(chart_hora, use_container_width=True)
    if not horas_bajas.empty:
        st.write(f"⏱️ Horas con baja demanda: {', '.join(horas_bajas['hora'].astype(str))}")
        st.info("Sugerencia: Active promociones dirigidas en estas horas para mejorar flujo.")

    # Clientes principales
    st.subheader("👥 Clientes Principales")
    df_vent_filt['cliente_id'] = pd.to_numeric(df_vent_filt['cliente_id'], errors='coerce').astype('Int64')
    top_clientes = df_vent_filt.groupby('cliente_id')['total'].sum().reset_index().sort_values('total', ascending=False).head(10)
    top_clientes = clientes.merge(top_clientes, left_on='id', right_on='cliente_id', how='inner')
    st.dataframe(top_clientes[['nombre', 'visitas_total', 'total']].rename(columns={'total': 'Gasto Total'}))

    # Churn
    st.subheader("📉 Churn de Clientes")
    churn = (clientes['visitas_total'] == 1).mean() * 100
    st.metric("Churn (%)", f"{churn:.1f}%")
    if churn > 30:
        st.warning("⚠️ Churn elevado. Considere activar programa de lealtad o promociones de reactivación.")

# Pestaña 2: Sociodemográfico
with tab2:
    st.header("🏘️ Información Sociodemográfica: Mercado Local")

    try:
        # Datos base
        ageb = socio['ageb'].iloc[0]
        habitantes = socio['habitantes'].iloc[0]
        por_jovenes = socio['por_edad_20_39'].iloc[0]
        consumo_prom = socio['consumo_promedio'].iloc[0]
        competidores = socio['negocios_similares'].iloc[0]

        st.markdown(f"**AGEB:** {ageb} | **Habitantes:** {habitantes:,}")
        st.markdown(f"**Población 20–39 años:** {por_jovenes:.1f}% | **Consumo promedio en cafés:** ${consumo_prom:,.0f} MXN/mes")
        st.markdown(f"**Negocios similares en la zona:** {competidores}")

        # Gráfico de edad
        chart_edad = alt.Chart(pd.DataFrame({'Edad': ['20–39'], 'Porcentaje': [por_jovenes]})).mark_bar().encode(
            x='Edad:N', y='Porcentaje:Q'
        ).properties(title="Población Joven en Zona")
        st.altair_chart(chart_edad, use_container_width=True)

        # Índice de oportunidad local
        oportunidad = (por_jovenes * consumo_prom) / (competidores + 1)
        nivel = "🟢 Alta" if oportunidad > 1000 else "🟡 Moderada" if oportunidad > 500 else "🔴 Saturada"
        st.metric("Índice de Oportunidad", f"{oportunidad:.0f}", help="Calculado con población joven, consumo y competencia")
        st.write(f"**Nivel de oportunidad:** {nivel}")

        # Segmentación por estilo de vida
        st.subheader("👥 Segmentación de Clientes")
        segmentos = clientes.groupby('preferencias')['id'].count().reset_index().rename(columns={'id': 'Clientes'})
        chart_segmentos = alt.Chart(segmentos).mark_bar().encode(
            x='preferencias:N', y='Clientes:Q', color='preferencias:N'
        ).properties(title="Preferencias de Consumo")
        st.altair_chart(chart_segmentos, use_container_width=True)

        # Mapa de calor de demanda
        st.subheader("📍 Mapa de Clientes y Competencia")
        m = folium.Map(location=[19.4, -99.16], zoom_start=13)
        folium.Marker([19.4, -99.16], popup="Café Integral", icon=folium.Icon(color='green')).add_to(m)

        # Clientes con gasto
        clientes_gasto = ventas.groupby('cliente_id')['total'].sum().reset_index()
        clientes_geo = clientes.merge(clientes_gasto, left_on='id', right_on='cliente_id', how='inner')
        for _, row in clientes_geo.iterrows():
            if pd.notnull(row['lat']) and pd.notnull(row['lon']):
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=min(row['total'] / 100, 10),
                    color='blue',
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"{row['nombre']} (${row['total']:.0f})"
                ).add_to(m)

        # Competidores
        for comp in eval(socio['competidores_geo'].iloc[0]):
            folium.Marker([comp['lat'], comp['lon']], popup="Competidor", icon=folium.Icon(color='orange')).add_to(m)

        st_folium(m, width=700)

        # Saturación competitiva
        st.subheader("📊 Saturación Competitiva")
        st.write(f"Cada cliente tiene en promedio **~{competidores // 5} opciones** en un radio de 1 km.")
        if competidores > 15:
            st.warning("⚠️ Alta saturación en la zona. Considere diferenciarse con experiencia, menú o fidelización.")

        # Simulador de expansión
        st.subheader("📍 Simulador de Expansión Geográfica")
        zona_nueva = st.selectbox("Selecciona zona para simular expansión", ["Narvarte", "Del Valle", "Roma Norte", "Condesa"])
        if zona_nueva:
            # Simulación ficticia (puede conectarse a INEGI o API externa)
            datos_zona = {
                "Narvarte": {"por_jovenes": 42, "consumo": 950, "competencia": 8},
                "Del Valle": {"por_jovenes": 38, "consumo": 1100, "competencia": 12},
                "Roma Norte": {"por_jovenes": 45, "consumo": 1300, "competencia": 18},
                "Condesa": {"por_jovenes": 50, "consumo": 1400, "competencia": 20}
            }
            z = datos_zona[zona_nueva]
            oportunidad_zona = (z['por_jovenes'] * z['consumo']) / (z['competencia'] + 1)
            nivel_zona = "🟢 Alta" if oportunidad_zona > 1000 else "🟡 Moderada" if oportunidad_zona > 500 else "🔴 Saturada"
            st.write(f"**Zona:** {zona_nueva} | Población joven: {z['por_jovenes']}% | Consumo: ${z['consumo']} | Competencia: {z['competencia']}")
            st.metric("Oportunidad proyectada", f"{oportunidad_zona:.0f}")
            st.write(f"**Nivel de oportunidad:** {nivel_zona}")
            if oportunidad_zona > oportunidad:
                st.success("✅ Esta zona tiene mayor potencial que la actual. Considere expansión o activación de marca.")
            else:
                st.info("ℹ️ La zona actual tiene mejor oportunidad. Refuerce fidelización y diferenciación.")

    except Exception as e:
        st.warning(f"Error en análisis sociodemográfico: {str(e)}")

# Pestaña 3: Análisis IA
with tab3:
    st.header("🤖 Análisis de IA: Estrategias Personalizadas")

    try:
        # Función para asegurar columna 'cluster'
        def asegurar_columna_cluster():
            conn = sqlite3.connect('integral360.db')
            c = conn.cursor()
            try:
                c.execute("SELECT cluster FROM clientes LIMIT 1")
            except sqlite3.OperationalError:
                c.execute("ALTER TABLE clientes ADD COLUMN cluster INTEGER DEFAULT -1")
            conn.commit()
            conn.close()

        # Ejecutar verificación
        asegurar_columna_cluster()

        # Validar existencia y formato de cliente_id
        if 'cliente_id' not in ventas.columns:
            st.warning("⚠️ La columna 'cliente_id' no está disponible en ventas.")
            st.stop()

        ventas['cliente_id'] = pd.to_numeric(ventas['cliente_id'], errors='coerce').astype('Int64')
        ventas = ventas.dropna(subset=['cliente_id'])
        ventas = ventas[ventas['cliente_id'].between(1, 100)]

        clientes['id'] = pd.to_numeric(clientes['id'], errors='coerce').astype('Int64')
        clientes = clientes.dropna(subset=['id'])

        # Merge limpio
        df_ia = ventas.merge(clientes, left_on='cliente_id', right_on='id', how='inner')
        df_ia = df_ia.merge(socio, how='cross')

        # Validar columnas necesarias
        required_cols = ['edad', 'total', 'por_edad_20_39', 'producto', 'costo_insumo']
        if not all(col in df_ia.columns for col in required_cols):
            st.warning(f"⚠️ Faltan columnas necesarias para análisis IA: {required_cols}")
            st.stop()

        # Clustering
        features = df_ia[['edad', 'total', 'por_edad_20_39']].fillna(0)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
        df_ia['cluster'] = kmeans.labels_

        # Asignar cluster por cliente
        cluster_por_cliente = df_ia.groupby('cliente_id')['cluster'].agg(lambda x: x.value_counts().idxmax()).reset_index()
        conn = sqlite3.connect('integral360.db')
        for _, row in cluster_por_cliente.iterrows():
            conn.execute("UPDATE clientes SET cluster=? WHERE id=?", (int(row['cluster']), int(row['cliente_id'])))
        conn.commit()
        conn.close()

        # Producto estrella por cluster
        estrella = df_ia.groupby(['cluster', 'producto'])['total'].sum().reset_index()
        top_prod = estrella.sort_values(['cluster', 'total'], ascending=False).groupby('cluster').first().reset_index()

        st.subheader("Clusters y Productos Estrella")
        st.dataframe(top_prod)

        # Costo promedio por producto
        costo_prom = df_ia.groupby('producto')['costo_insumo'].mean().reset_index().rename(columns={'costo_insumo': 'Costo Promedio'})
        st.subheader("Costo Promedio por Producto")
        st.dataframe(costo_prom)

        # Visualización
        chart_cluster = alt.Chart(df_ia).mark_circle().encode(
            x='edad:Q',
            y='total:Q',
            color='cluster:N',
            tooltip=['producto', 'total', 'cluster']
        ).properties(title="Clusters Clientes vs. Zona")
        st.altair_chart(chart_cluster, use_container_width=True)

        # Definición de clusters
        st.subheader("🧠 Definición de Clusters Detectados")
        cluster_def = df_ia.groupby('cluster').agg({
            'edad': 'mean',
            'total': 'mean',
            'por_edad_20_39': 'mean'
        }).reset_index()

        def interpretar_cluster(row):
            if row['edad'] < 30 and row['total'] < 100:
                return "🎓 Jóvenes estudiantes con consumo moderado"
            elif row['edad'] >= 30 and row['total'] > 150:
                return "💼 Profesionales con alto poder adquisitivo"
            else:
                return "🎨 Freelancers o creativos con consumo variable"

        cluster_def['Perfil'] = cluster_def.apply(interpretar_cluster, axis=1)
        st.dataframe(cluster_def.rename(columns={
            'cluster': 'Cluster',
            'edad': 'Edad Promedio',
            'total': 'Consumo Promedio',
            'por_edad_20_39': '% Zona Joven'
        }))

        # Promociones inteligentes
        st.subheader("📈 Promociones Inteligentes por Cluster")
        promociones = []
        for _, row in top_prod.iterrows():
            producto = row['producto']
            cluster = row['cluster']
            base_ventas = df_ia[(df_ia['producto'] == producto) & (df_ia['cluster'] == cluster)]['total'].sum()
            costo_unitario = df_ia[df_ia['producto'] == producto]['costo_insumo'].mean()
            incremento_estimado = base_ventas * 0.18
            gasto_extra = costo_unitario * 0.10 * len(df_ia[df_ia['cluster'] == cluster])

            promociones.append({
                'Cluster': cluster,
                'Producto Estrella': producto,
                'Incremento Esperado ($)': round(incremento_estimado, 2),
                'Gasto Extra Estimado ($)': round(gasto_extra, 2),
                'Tipo de Promoción': f"2x1 en {producto}" if costo_unitario < 30 else f"Combo {producto} + snack"
            })

        df_promos = pd.DataFrame(promociones).sort_values('Incremento Esperado ($)', ascending=False).head(5)
        st.dataframe(df_promos)

        st.write("**Estrategia sugerida**: Aplicar estas 5 promociones durante las próximas dos semanas. Están optimizadas para generar ingresos sin elevar significativamente el gasto operativo.")

    except Exception as e:
        st.warning(f"Error en análisis de IA: {str(e)}")

# Pestaña 4: Predictivo
with tab4:
    st.header("🔮 Análisis Predictivo: Simula Combos y Promociones")

    try:
        # Selección múltiple de productos
        productos_combo = st.multiselect("Selecciona productos para el combo", ventas['producto'].unique())

        if not productos_combo:
            st.info("Selecciona al menos un producto para simular el combo.")
            st.stop()

        # Filtro de ventas por productos seleccionados
        df_combo = ventas[ventas['producto'].isin(productos_combo)].copy()

        if df_combo.empty:
            st.warning("No hay datos de ventas para los productos seleccionados.")
            st.stop()

        # Agrupamiento por cliente y producto
        combo_por_cliente = df_combo.groupby(['cliente_id', 'producto'])['cantidad'].sum().unstack(fill_value=0)

        # Clustering no supervisado
        kmeans = KMeans(n_clusters=3, random_state=42).fit(combo_por_cliente)
        combo_por_cliente['cluster'] = kmeans.labels_

        # Simulación de impacto promocional
        promo_impacto = st.slider("Impacto promocional estimado (% de incremento en clientes)", 0, 50, 20)

        ingresos_actuales = df_combo.groupby('cliente_id')['total'].sum().groupby(combo_por_cliente['cluster']).sum()
        ingresos_proyectados = ingresos_actuales * (1 + promo_impacto / 100)

        # Visualización comparativa
        df_ingresos = pd.DataFrame({
            'Cluster': ingresos_actuales.index,
            'Actual': ingresos_actuales.values,
            'Proyectado': ingresos_proyectados.values
        })

        chart = alt.Chart(df_ingresos.melt('Cluster')).mark_bar().encode(
            x='Cluster:N',
            y='value:Q',
            color='variable:N',
            tooltip=['Cluster', 'variable', 'value']
        ).properties(title="Ingresos por Combo y Cluster")
        st.altair_chart(chart, use_container_width=True)

        # Sugerencias de combos por cluster
        st.subheader("🧠 Combos sugeridos por cluster")
        sugerencias = combo_por_cliente.groupby('cluster').mean().apply(lambda x: x[x > 0.5].index.tolist(), axis=1)

        combos = []
        for cluster, productos in sugerencias.items():
            ingreso = ingresos_proyectados[cluster]
            gasto_estimado = df_combo[df_combo['producto'].isin(productos)]['costo_insumo'].mean() * len(productos) * 0.1
            combos.append({
                'Cluster': cluster,
                'Combo sugerido': ', '.join(productos),
                'Ingreso proyectado ($)': round(ingreso, 2),
                'Gasto estimado ($)': round(gasto_estimado, 2),
                'Rentabilidad (%)': round((ingreso - gasto_estimado) / ingreso * 100, 1) if ingreso > 0 else 0
            })

        st.dataframe(pd.DataFrame(combos))

        st.write("**Recomendación institucional**: Aplicar combos sugeridos por cluster durante las próximas dos semanas. Están optimizados para generar ingresos sin elevar significativamente el gasto operativo.")
    except Exception as e:
        st.warning(f"Error en análisis predictivo: {str(e)}")

# Pestaña 5: Lealtad Clientes
with tab5:
    st.header("❤️ Programa de Lealtad")

    try:
        # Validar columnas
        if 'id' not in clientes.columns or 'cliente_id' not in ventas.columns:
            st.warning("⚠️ Las columnas necesarias para el análisis no están disponibles.")
            st.stop()

        # Preparar datos para clustering
        ventas['cliente_id'] = pd.to_numeric(ventas['cliente_id'], errors='coerce').astype('Int64')
        clientes['id'] = pd.to_numeric(clientes['id'], errors='coerce').astype('Int64')

        df_ia = ventas.merge(clientes, left_on='cliente_id', right_on='id', how='inner').merge(socio, how='cross')

        # Clustering
        features = df_ia[['edad', 'total', 'por_edad_20_39']].fillna(0)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(features)
        df_ia['cluster'] = kmeans.labels_

        # Asignar cluster por cliente
        cluster_por_cliente = df_ia.groupby('cliente_id')['cluster'].agg(lambda x: x.value_counts().idxmax()).reset_index()
        clientes_cluster = clientes.merge(cluster_por_cliente, left_on='id', right_on='cliente_id', how='left')
        clientes_cluster['cluster'] = clientes_cluster['cluster'].fillna(-1).astype(int)

        # Mostrar tabla con cluster
        st.subheader("📋 Clientes y Clusters")
        st.dataframe(clientes_cluster[['id', 'nombre', 'puntos_lealtad', 'ultima_visita', 'cluster']].rename(columns={'cluster': 'Cluster'}))

        # Generar QR para nuevo cliente
        if st.button("Generar QR para Cliente Nuevo"):
            new_id = len(clientes) + 1
            qr = qrcode.QRCode()
            qr.add_data(f"ID:{new_id}")
            img = qr.make_image()
            bio = BytesIO()
            img.save(bio, 'PNG')
            st.image(bio.getvalue(), caption=f"QR para ID {new_id}")

            new_cliente = pd.DataFrame([{
                'id': new_id,
                'nombre': f'Cliente {new_id}',
                'email': f'cliente{new_id}@ejemplo.com',
                'telefono': f'55{new_id:07d}',
                'direccion': 'Roma Norte, CDMX',
                'edad': 30,
                'sexo': 'Otro',
                'visitas_total': 0,
                'preferencias': 'Dulce',
                'puntos_lealtad': 0,
                'ultima_visita': datetime.now().strftime('%Y-%m-%d'),
                'qr_code': f'qr_{new_id}.png',
                'lat': 19.42,
                'lon': -99.16
            }])
            new_cliente.to_sql('clientes', sqlite3.connect('integral360.db'), if_exists='append', index=False)
            st.success("✅ Cliente nuevo registrado con QR.")
    except Exception as e:
        st.warning(f"Error en programa de lealtad: {str(e)}")
