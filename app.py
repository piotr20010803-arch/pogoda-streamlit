import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Pogoda i jakosc powietrza w Polsce", layout="wide")

# ---------------------------------------------------------------------------
# Dane wejsciowe - 10 miast wojewodzkich (prawdziwe wspolrzedne)
# ---------------------------------------------------------------------------

MIASTA = {
    "Warszawa":  (52.2297, 21.0122),
    "Krakow":    (50.0647, 19.9450),
    "Wroclaw":   (51.1079, 17.0385),
    "Poznan":    (52.4064, 16.9252),
    "Gdansk":    (54.3520, 18.6466),
    "Lodz":      (51.7592, 19.4560),
    "Katowice":  (50.2649, 19.0238),
    "Lublin":    (51.2465, 22.5684),
    "Bialystok": (53.1325, 23.1688),
    "Szczecin":  (53.4285, 14.5528),
}

KOLORY_MIAST = dict(zip(MIASTA.keys(), px.colors.qualitative.Set1 + px.colors.qualitative.Set2))

PROGI_AQI = [0, 20, 40, 60, 80, 100, 1000]
ETYKIETY_AQI = ["Bardzo dobra", "Dobra", "Umiarkowana", "Zla", "Bardzo zla", "Ekstremalnie zla"]


# ---------------------------------------------------------------------------
# Pobieranie danych - Open-Meteo (bez klucza API)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Pobieranie danych pogodowych z Open-Meteo...")
def pobierz_pogode(miasta, dni_wstecz):
    ramki = []
    for nazwa, (lat, lon) in miasta.items():
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                    "past_days": dni_wstecz,
                    "forecast_days": 1,
                    "timezone": "Europe/Warsaw",
                },
                timeout=20,
            )
            r.raise_for_status()
            dane = r.json()["daily"]
        except (requests.RequestException, KeyError):
            continue
        df = pd.DataFrame(dane)
        df["miasto"] = nazwa
        df["latitude"] = lat
        df["longitude"] = lon
        ramki.append(df)
    if not ramki:
        return pd.DataFrame()
    return pd.concat(ramki, ignore_index=True)


@st.cache_data(ttl=3600, show_spinner="Pobieranie danych o jakosci powietrza z Open-Meteo...")
def pobierz_jakosc_powietrza(miasta, dni_wstecz):
    ramki = []
    for nazwa, (lat, lon) in miasta.items():
        try:
            r = requests.get(
                "https://air-quality-api.open-meteo.com/v1/air-quality",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "pm10,pm2_5,ozone,nitrogen_dioxide,european_aqi",
                    "past_days": dni_wstecz,
                    "forecast_days": 1,
                    "timezone": "Europe/Warsaw",
                },
                timeout=20,
            )
            r.raise_for_status()
            dane = r.json()["hourly"]
        except (requests.RequestException, KeyError):
            continue
        godzinowe = pd.DataFrame(dane)
        godzinowe["czas"] = pd.to_datetime(godzinowe["time"])
        godzinowe["time"] = godzinowe["czas"].dt.date.astype(str)
        dzienne = godzinowe.groupby("time").agg(
            pm10=("pm10", "mean"),
            pm2_5=("pm2_5", "mean"),
            ozon=("ozone", "mean"),
            no2=("nitrogen_dioxide", "mean"),
            aqi=("european_aqi", "mean"),
        ).reset_index()
        dzienne["miasto"] = nazwa
        dzienne["latitude"] = lat
        dzienne["longitude"] = lon
        ramki.append(dzienne)
    if not ramki:
        return pd.DataFrame()
    return pd.concat(ramki, ignore_index=True)


def przygotuj_dane(dni_wstecz):
    pogoda = pobierz_pogode(MIASTA, dni_wstecz)
    jakosc = pobierz_jakosc_powietrza(MIASTA, dni_wstecz)

    if pogoda.empty or jakosc.empty:
        return pd.DataFrame()

    pogoda = pogoda.rename(columns={"time": "data"})
    jakosc = jakosc.rename(columns={"time": "data"})

    df = pd.merge(
        pogoda, jakosc,
        on=["miasto", "data", "latitude", "longitude"],
        how="inner",
    )

    df["data"] = pd.to_datetime(df["data"])

    # czyszczenie - usuwamy dni z brakujacymi kluczowymi pomiarami
    df = df.dropna(subset=["temperature_2m_max", "temperature_2m_min", "aqi"])

    # konwersje typow i kolumny pochodne
    df["temp_srednia"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
    df["zakres_temperatur"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    df["aqi"] = df["aqi"].round(0)
    df["kategoria_aqi"] = pd.cut(df["aqi"], bins=PROGI_AQI, labels=ETYKIETY_AQI)

    df = df.rename(columns={
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "opady_mm",
        "wind_speed_10m_max": "wiatr_kmh",
    })

    kolejnosc = [
        "miasto", "data", "latitude", "longitude",
        "temp_min", "temp_max", "temp_srednia", "zakres_temperatur",
        "opady_mm", "wiatr_kmh",
        "pm10", "pm2_5", "ozon", "no2", "aqi", "kategoria_aqi",
    ]
    return df[kolejnosc].sort_values(["miasto", "data"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sidebar - filtry
# ---------------------------------------------------------------------------

st.sidebar.header("Filtry")

dni_wstecz = st.sidebar.slider(
    "Liczba dni historii do pobrania", min_value=14, max_value=90, value=60, step=1,
)

df = przygotuj_dane(dni_wstecz)

if df.empty:
    st.error("Nie udalo sie pobrac danych z Open-Meteo. Sprobuj odswiezyc strone za chwile.")
    st.stop()

wszystkie_miasta = sorted(df["miasto"].unique())
wybrane_miasta = st.sidebar.multiselect(
    "Miasta", options=wszystkie_miasta, default=wszystkie_miasta,
)

min_data, max_data = df["data"].min().date(), df["data"].max().date()
zakres_dat = st.sidebar.date_input(
    "Zakres dat", value=(min_data, max_data), min_value=min_data, max_value=max_data,
)

wskaznik = st.sidebar.selectbox(
    "Wskaznik jakosci powietrza do analizy",
    options=["pm2_5", "pm10", "ozon", "no2", "aqi"],
    index=0,
    format_func=lambda x: {"pm2_5": "PM2.5", "pm10": "PM10", "ozon": "Ozon", "no2": "NO2", "aqi": "AQI"}[x],
)

if not wybrane_miasta:
    st.warning("Wybierz przynajmniej jedno miasto w panelu bocznym.")
    st.stop()

if isinstance(zakres_dat, tuple) and len(zakres_dat) == 2:
    data_od, data_do = zakres_dat
else:
    data_od, data_do = min_data, max_data

dff = df[
    df["miasto"].isin(wybrane_miasta)
    & (df["data"].dt.date >= data_od)
    & (df["data"].dt.date <= data_do)
].copy()

if dff.empty:
    st.warning("Brak danych dla wybranych filtrow.")
    st.stop()

mapa_kolorow = {m: KOLORY_MIAST[m] for m in wybrane_miasta}


# ---------------------------------------------------------------------------
# Naglowek i KPI
# ---------------------------------------------------------------------------

st.title("Pogoda i jakosc powietrza w polskich miastach")
st.caption(
    "Dane pobierane na biezaco z darmowego API Open-Meteo (bez klucza) - "
    "prognoza+historia pogody oraz jakosc powietrza (PM10, PM2.5, ozon, NO2, europejski indeks AQI)."
)

kol1, kol2, kol3, kol4 = st.columns(4)
kol1.metric("Srednia temperatura", f"{dff['temp_srednia'].mean():.1f} C")
kol2.metric("Suma opadow (srednio/miasto)", f"{dff.groupby('miasto')['opady_mm'].sum().mean():.0f} mm")
kol3.metric("Srednie AQI", f"{dff['aqi'].mean():.0f}")
najgorsze_miasto = dff.groupby("miasto")["aqi"].mean().idxmax()
kol4.metric("Najgorsze powietrze", najgorsze_miasto)

st.divider()


# ---------------------------------------------------------------------------
# Zakladki z wykresami
# ---------------------------------------------------------------------------

tab_przeglad, tab_pogoda, tab_powietrze, tab_zaleznosci, tab_dane = st.tabs(
    ["Przeglad i mapa", "Pogoda", "Jakosc powietrza", "Zaleznosci", "Dane"]
)

with tab_przeglad:
    st.subheader("Srednie warunki wg miasta")
    agregat = dff.groupby(["miasto", "latitude", "longitude"]).agg(
        srednia_temp=("temp_srednia", "mean"),
        srednie_aqi=("aqi", "mean"),
        suma_opadow=("opady_mm", "sum"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig_mapa = px.scatter_map(
            agregat, lat="latitude", lon="longitude",
            size="srednie_aqi", color="srednia_temp",
            hover_name="miasto",
            hover_data={"srednia_temp": ":.1f", "srednie_aqi": ":.0f", "suma_opadow": ":.0f",
                        "latitude": False, "longitude": False},
            color_continuous_scale="RdYlBu_r",
            zoom=4.3, height=500,
            title="Srednia temperatura (kolor) i AQI (rozmiar) wg miasta",
        )
        fig_mapa.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_mapa, width="stretch")

    with c2:
        agregat_sort = agregat.sort_values("srednie_aqi", ascending=False)
        fig_bar = px.bar(
            agregat_sort, x="miasto", y="srednie_aqi", color="miasto",
            color_discrete_map=mapa_kolorow,
            title="Srednie AQI wg miasta (im wyzej, tym gorsze powietrze)",
            labels={"miasto": "Miasto", "srednie_aqi": "Srednie AQI"},
        )
        fig_bar.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_bar, width="stretch")

    st.caption(
        f"W wybranym okresie najgorsza srednia jakosc powietrza wystapila w miescie "
        f"**{najgorsze_miasto}**. Miasta o wiekszej srednicy punktu na mapie maja wyzsze AQI."
    )

with tab_pogoda:
    st.subheader("Temperatura w czasie")
    fig_linia_temp = px.line(
        dff, x="data", y="temp_srednia", color="miasto",
        color_discrete_map=mapa_kolorow,
        title="Srednia dobowa temperatura wg miasta",
        labels={"data": "Data", "temp_srednia": "Temperatura (C)", "miasto": "Miasto"},
    )
    st.plotly_chart(fig_linia_temp, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig_box_temp = px.box(
            dff, x="miasto", y="temp_srednia", color="miasto",
            color_discrete_map=mapa_kolorow,
            title="Rozklad temperatur wg miasta",
            labels={"miasto": "Miasto", "temp_srednia": "Temperatura (C)"},
        )
        fig_box_temp.update_layout(showlegend=False)
        st.plotly_chart(fig_box_temp, width="stretch")
    with c2:
        fig_opady = px.bar(
            dff.groupby("miasto")["opady_mm"].sum().reset_index(),
            x="miasto", y="opady_mm", color="miasto",
            color_discrete_map=mapa_kolorow,
            title="Suma opadow w wybranym okresie",
            labels={"miasto": "Miasto", "opady_mm": "Opady (mm)"},
        )
        fig_opady.update_layout(showlegend=False)
        st.plotly_chart(fig_opady, width="stretch")

    st.caption(
        "Szerokosc pudelka boxplotu pokazuje zmiennosc temperatur w danym miescie - "
        "miasta nadmorskie (np. Gdansk, Szczecin) zwykle maja mniejsze wahania niz miasta w glebi kraju."
    )

with tab_powietrze:
    etykieta = {"pm2_5": "PM2.5", "pm10": "PM10", "ozon": "Ozon", "no2": "NO2", "aqi": "AQI"}[wskaznik]

    st.subheader(f"{etykieta} w czasie")
    fig_linia_aq = px.line(
        dff, x="data", y=wskaznik, color="miasto",
        color_discrete_map=mapa_kolorow,
        title=f"{etykieta} w czasie wg miasta",
        labels={"data": "Data", wskaznik: etykieta, "miasto": "Miasto"},
    )
    st.plotly_chart(fig_linia_aq, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(
            dff, x=wskaznik, color="miasto", nbins=30,
            color_discrete_map=mapa_kolorow,
            title=f"Rozklad wartosci {etykieta}",
            labels={wskaznik: etykieta},
        )
        st.plotly_chart(fig_hist, width="stretch")
    with c2:
        rozklad_kategorii = dff["kategoria_aqi"].value_counts().reindex(ETYKIETY_AQI).fillna(0).reset_index()
        rozklad_kategorii.columns = ["kategoria", "liczba_dni"]
        fig_kategorie = px.bar(
            rozklad_kategorii, x="kategoria", y="liczba_dni",
            title="Liczba dni w poszczegolnych kategoriach AQI (wszystkie miasta)",
            labels={"kategoria": "Kategoria AQI", "liczba_dni": "Liczba dni"},
        )
        st.plotly_chart(fig_kategorie, width="stretch")

    st.caption(
        "Europejski indeks AQI: 0-20 bardzo dobra, 20-40 dobra, 40-60 umiarkowana, "
        "60-80 zla, 80-100 bardzo zla, powyzej 100 ekstremalnie zla jakosc powietrza."
    )

with tab_zaleznosci:
    st.subheader("Czy cieplejsze dni maja czystsze powietrze?")
    fig_scatter = px.scatter(
        dff, x="temp_srednia", y=wskaznik, color="miasto",
        size="wiatr_kmh", hover_data=["data"],
        color_discrete_map=mapa_kolorow,
        title=f"Temperatura vs {etykieta} (rozmiar punktu = predkosc wiatru)",
        labels={"temp_srednia": "Srednia temperatura (C)", wskaznik: etykieta},
    )
    st.plotly_chart(fig_scatter, width="stretch")

    korelacja = dff["temp_srednia"].corr(dff[wskaznik])

    kolumny_numeryczne = ["temp_srednia", "zakres_temperatur", "opady_mm", "wiatr_kmh",
                           "pm10", "pm2_5", "ozon", "no2", "aqi"]
    macierz = dff[kolumny_numeryczne].corr()
    fig_heatmap = px.imshow(
        macierz, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Macierz korelacji miedzy zmiennymi pogodowymi a jakoscia powietrza",
    )
    st.plotly_chart(fig_heatmap, width="stretch")

    st.caption(
        f"Korelacja miedzy srednia temperatura a {etykieta} w wybranym okresie i miastach wynosi "
        f"**{korelacja:.2f}**. Wartosc bliska 0 oznacza brak wyraznego zwiazku liniowego, "
        f"wartosc ujemna sugeruje, ze cieplejszym dniom towarzyszy nizsze zanieczyszczenie "
        f"(czesciej zwiazane z wiatrem i opadami niz z sama temperatura)."
    )

with tab_dane:
    st.subheader("Dane po czyszczeniu i przygotowaniu (przefiltrowane)")
    st.dataframe(dff, width="stretch")
    st.download_button(
        "Pobierz przefiltrowane dane jako CSV",
        data=dff.to_csv(index=False).encode("utf-8"),
        file_name="pogoda_jakosc_powietrza.csv",
        mime="text/csv",
    )
