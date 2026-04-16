import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="ScoutIQ", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Bebas+Neue&display=swap');
:root{--primary:#E63946;--secondary:#1D3557;--accent:#457B9D;--gold:#F4A261;--green:#2A9D8F;--bg:#0A0E1A;--card:#111827;--border:#1E2D40;--text:#E8EDF5;--muted:#8B9DC3;}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif;background-color:var(--bg);color:var(--text);}
.stApp{background-color:var(--bg);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1421 0%,#111827 100%);border-right:1px solid var(--border);}
[data-testid="metric-container"]{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1rem;transition:transform 0.2s;}
[data-testid="metric-container"]:hover{transform:translateY(-2px);}
h1{font-family:'Bebas Neue',sans-serif;letter-spacing:2px;color:var(--text);}
h2{font-family:'Bebas Neue',sans-serif;letter-spacing:1px;color:var(--accent);}
h3{font-weight:600;color:var(--text);}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:0.75rem!important;text-transform:uppercase;letter-spacing:1px;}
[data-testid="stMetricValue"]{color:var(--text)!important;font-weight:700;font-size:1.6rem!important;}
[data-testid="stMetricDelta"]{color:var(--green)!important;}
.badge-sb{background:#2A9D8F22;color:#2A9D8F;border:1px solid #2A9D8F;border-radius:20px;padding:3px 12px;font-weight:700;font-size:0.75rem;display:inline-block;}
.badge-buy{background:#E9C46A22;color:#E9C46A;border:1px solid #E9C46A;border-radius:20px;padding:3px 12px;font-weight:700;font-size:0.75rem;display:inline-block;}
.badge-mon{background:#F4A26122;color:#F4A261;border:1px solid #F4A261;border-radius:20px;padding:3px 12px;font-weight:700;font-size:0.75rem;display:inline-block;}
.badge-pass{background:#ADB5BD22;color:#ADB5BD;border:1px solid #ADB5BD;border-radius:20px;padding:3px 12px;font-weight:700;font-size:0.75rem;display:inline-block;}
.player-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:1.5rem;margin-bottom:1rem;transition:border-color 0.2s;}
.player-card:hover{border-color:var(--primary);}
.section-header{background:linear-gradient(90deg,#E6394622,transparent);border-left:4px solid var(--primary);padding:0.75rem 1.25rem;border-radius:0 8px 8px 0;margin-bottom:1.5rem;}
hr{border-color:var(--border);margin:2rem 0;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
</style>
""", unsafe_allow_html=True)

def _read_csv(path):
    """Handle CSVs where every row is wrapped in outer double-quotes."""
    import io
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        cleaned.append(line)
    df = pd.read_csv(io.StringIO('\n'.join(cleaned)))
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_data():
    master    = _read_csv('scoutiq_master.csv')
    complete  = _read_csv('scoutiq_complete.csv')
    salah_fin = _read_csv('scoutiq_salah_financial.csv')
    kdb_fin   = _read_csv('scoutiq_kdb_financial.csv')
    return master, complete, salah_fin, kdb_fin

master, complete, salah_fin, kdb_fin = load_data()

features = ['Gls_90','Ast_90','GA_90','Sh_90','SoT_90','SoT_pct','G_per_Sh']
salah_bench = {'Gls_90':0.505,'Ast_90':0.396,'GA_90':0.901,'Sh_90':2.307,'SoT_90':0.973,'SoT_pct':42.19,'G_per_Sh':0.219}
kdb_bench   = {'Gls_90':0.305,'Ast_90':0.610,'GA_90':0.915,'Sh_90':2.927,'SoT_90':1.250,'SoT_pct':42.71,'G_per_Sh':0.104}
colors      = {'Eredivisie':'#E63946','Primeira Liga':'#2A9D8F','EFL Championship':'#E9C46A'}

PLOTLY_LAYOUT = dict(
    plot_bgcolor='#111827', paper_bgcolor='#111827',
    font=dict(color='#E8EDF5', family='Space Grotesk'),
    title_font=dict(size=15, color='#E8EDF5'),
    legend=dict(bgcolor='#111827', bordercolor='#1E2D40', borderwidth=1)
)

def badge_html(rec):
    cls = {'STRONG BUY':'badge-sb','BUY':'badge-buy','MONITOR':'badge-mon','PASS':'badge-pass'}.get(rec,'badge-pass')
    return f'<span class="{cls}">{rec}</span>'

def make_radar(player_vals, player_name, bench_vals, bench_name, bench_color):
    scaler = MinMaxScaler()
    all_v = pd.DataFrame([player_vals, bench_vals], columns=features)
    norm  = scaler.fit_transform(all_v)
    p_n   = list(norm[0]) + [norm[0][0]]
    b_n   = list(norm[1]) + [norm[1][0]]
    theta = features + [features[0]]
    fig   = go.Figure()
    fig.add_trace(go.Scatterpolar(r=b_n, theta=theta, fill='toself', name=bench_name, line_color=bench_color, opacity=0.4))
    fig.add_trace(go.Scatterpolar(r=p_n, theta=theta, fill='toself', name=player_name, line_color='#E63946'))
    fig.update_layout(**PLOTLY_LAYOUT, height=380,
        polar=dict(bgcolor='#0D1421',
                   radialaxis=dict(visible=True, range=[0,1], color='#4A5568', gridcolor='#1E2D40'),
                   angularaxis=dict(color='#8B9DC3', gridcolor='#1E2D40')),
        showlegend=True, margin=dict(l=40,r=40,t=40,b=40))
    return fig

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:1rem 0;'>
        <div style='font-family:Bebas Neue;font-size:2.5rem;color:#E63946;letter-spacing:3px;'>⚽ SCOUTIQ</div>
        <div style='color:#8B9DC3;font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;'>Find The Next Salah</div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Nav", ["🏠 Home","📊 Benchmarks","🔍 Scout Engine","👤 Player Profile","💹 Market Intelligence","📦 About"], label_visibility="collapsed")
    st.divider()
    st.markdown("<div style='color:#4A5568;font-size:0.7rem;text-align:center;'>Data: FBref + Transfermarkt<br>Season: 2023/24<br>NED · POR · ENG</div>", unsafe_allow_html=True)

# ════════════════ HOME ════════════════
if page == "🏠 Home":
    st.markdown("<h1 style='font-size:3.5rem;margin-bottom:0;'>SCOUTIQ</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8B9DC3;font-size:1.1rem;margin-top:0;margin-bottom:2rem;'>Find the Next Mo Salah. Before the Market Does.</p>", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    salah_rows = master[master['Benchmark']=='Mo Salah']
    kdb_rows   = master[master['Benchmark']=='Kevin De Bruyne']
    c1.metric("Players Analysed","264","3 leagues")
    c2.metric("Players Valued","34","Transfermarkt")
    c3.metric("STRONG BUY", str(len(master[master['Recommendation']=='STRONG BUY'])), "recommendations")
    c4.metric("Max Arbitrage", f"€{master['Arbitrage_M'].max():.1f}M","per signing")
    c5.metric("Top Similarity", f"{master['Similarity_Score'].max():.1%}","vs benchmark")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'><h3 style='margin:0;'>🔴 Top 5 — Next Mo Salah</h3></div>", unsafe_allow_html=True)
        for _, row in salah_rows.nlargest(5,'Decision_Score').iterrows():
            st.markdown(f"""<div class='player-card'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <div style='font-weight:700;font-size:1rem;'>{row['Player']}</div>
                        <div style='color:#8B9DC3;font-size:0.8rem;'>{row['Squad']} · {row['League']} · Age {int(row['Age'])}</div>
                    </div>
                    <div style='text-align:right;'>
                        {badge_html(row['Recommendation'])}
                        <div style='color:#E63946;font-weight:700;margin-top:4px;'>{row['Similarity_Score']:.1%} match</div>
                        <div style='color:#8B9DC3;font-size:0.75rem;'>€{row['Market_Value_M']:.1f}M · Saves €{row['Arbitrage_M']:.1f}M</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-header'><h3 style='margin:0;'>🔵 Top 5 — Next Kevin De Bruyne</h3></div>", unsafe_allow_html=True)
        for _, row in kdb_rows.nlargest(5,'Decision_Score').iterrows():
            st.markdown(f"""<div class='player-card'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <div style='font-weight:700;font-size:1rem;'>{row['Player']}</div>
                        <div style='color:#8B9DC3;font-size:0.8rem;'>{row['Squad']} · {row['League']} · Age {int(row['Age'])}</div>
                    </div>
                    <div style='text-align:right;'>
                        {badge_html(row['Recommendation'])}
                        <div style='color:#457B9D;font-weight:700;margin-top:4px;'>{row['Similarity_Score']:.1%} match</div>
                        <div style='color:#8B9DC3;font-size:0.75rem;'>€{row['Market_Value_M']:.1f}M · Saves €{row['Arbitrage_M']:.1f}M</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

# ════════════════ BENCHMARKS ════════════════
elif page == "📊 Benchmarks":
    st.markdown("<h1>BENCHMARK PROFILES</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8B9DC3;'>The two elite profiles ScoutIQ searches for in your target leagues.</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class='player-card' style='border-color:#E63946;'>
            <div style='font-family:Bebas Neue;font-size:1.8rem;color:#E63946;'>🔴 MOHAMED SALAH</div>
            <div style='color:#8B9DC3;'>AS Roma · 2015/16 · Age 23</div><hr style='border-color:#1E2D40;'>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;'>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>Goals/90</div><div style='font-size:1.4rem;font-weight:700;color:#E63946;'>0.505</div></div>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>Assists/90</div><div style='font-size:1.4rem;font-weight:700;'>0.396</div></div>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>Shots/90</div><div style='font-size:1.4rem;font-weight:700;'>2.307</div></div>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>SoT/90</div><div style='font-size:1.4rem;font-weight:700;'>0.973</div></div>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>Shot Acc.</div><div style='font-size:1.4rem;font-weight:700;'>42.19%</div></div>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>Market Val.</div><div style='font-size:1.4rem;font-weight:700;color:#F4A261;'>€15M</div></div>
            </div>
            <div style='margin-top:1rem;color:#8B9DC3;font-size:0.8rem;'>→ Sold 12 months later for £36.9M (+146%)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='player-card' style='border-color:#457B9D;'>
            <div style='font-family:Bebas Neue;font-size:1.8rem;color:#457B9D;'>🔵 KEVIN DE BRUYNE</div>
            <div style='color:#8B9DC3;'>VfL Wolfsburg · 2014/15 · Age 23</div><hr style='border-color:#1E2D40;'>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;'>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>Goals/90</div><div style='font-size:1.4rem;font-weight:700;'>0.305</div></div>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>Assists/90</div><div style='font-size:1.4rem;font-weight:700;color:#457B9D;'>0.610</div></div>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>Shots/90</div><div style='font-size:1.4rem;font-weight:700;'>2.927</div></div>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>SoT/90</div><div style='font-size:1.4rem;font-weight:700;'>1.250</div></div>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>Shot Acc.</div><div style='font-size:1.4rem;font-weight:700;'>42.71%</div></div>
                <div><div style='color:#8B9DC3;font-size:0.7rem;text-transform:uppercase;'>Market Val.</div><div style='font-size:1.4rem;font-weight:700;color:#F4A261;'>€18M</div></div>
            </div>
            <div style='margin-top:1rem;color:#8B9DC3;font-size:0.8rem;'>→ Sold to Man City for £55M (+206%)</div>
        </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("<h2>PROFILE COMPARISON</h2>", unsafe_allow_html=True)
    scaler2 = MinMaxScaler()
    all_b = pd.DataFrame([salah_bench, kdb_bench])
    norm2 = scaler2.fit_transform(all_b)
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatterpolar(r=list(norm2[0])+[norm2[0][0]], theta=features+[features[0]], fill='toself', name='Salah 2015/16', line_color='#E63946'))
    fig_b.add_trace(go.Scatterpolar(r=list(norm2[1])+[norm2[1][0]], theta=features+[features[0]], fill='toself', name='KDB 2014/15',   line_color='#457B9D'))
    fig_b.update_layout(**PLOTLY_LAYOUT, height=500, polar=dict(bgcolor='#0D1421', radialaxis=dict(visible=True,range=[0,1],color='#4A5568',gridcolor='#1E2D40'), angularaxis=dict(color='#8B9DC3',gridcolor='#1E2D40')))
    st.plotly_chart(fig_b, use_container_width=True)

# ════════════════ SCOUT ENGINE ════════════════
elif page == "🔍 Scout Engine":
    st.markdown("<h1>SCOUT ENGINE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8B9DC3;'>Filter and discover players by benchmark, league, budget and recommendation tier.</p>", unsafe_allow_html=True)
    f1,f2,f3,f4 = st.columns(4)
    bench_sel  = f1.selectbox("Benchmark", ["Mo Salah","Kevin De Bruyne"])
    leagues_sel= f2.multiselect("League", ["Eredivisie","Primeira Liga","EFL Championship"], default=["Eredivisie","Primeira Liga","EFL Championship"])
    max_val    = f3.slider("Max Market Value (€M)", 0.5, 25.0, 10.0, 0.5)
    recs_sel   = f4.multiselect("Recommendation", ["STRONG BUY","BUY","MONITOR","PASS"], default=["STRONG BUY","BUY"])
    df_f = master[(master['Benchmark']==bench_sel)&(master['League'].isin(leagues_sel))&(master['Market_Value_M']<=max_val)&(master['Recommendation'].isin(recs_sel))].sort_values('Decision_Score',ascending=False)
    st.markdown(f"<p style='color:#8B9DC3;'>Showing <b style='color:#E8EDF5;'>{len(df_f)}</b> players</p>", unsafe_allow_html=True)
    if len(df_f)==0:
        st.warning("No players match. Try relaxing filters.")
    else:
        fig_bar = px.bar(df_f.head(15), x='Similarity_Score', y='Player', color='League', color_discrete_map=colors, orientation='h', text='Similarity_Score', hover_data=['Squad','Age','Market_Value_M','Arbitrage_M','Recommendation'])
        fig_bar.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=430, yaxis=dict(autorange='reversed'), xaxis_title=f'Similarity vs {bench_sel}', xaxis=dict(gridcolor='#1E2D40'))
        st.plotly_chart(fig_bar, use_container_width=True)
        disp = df_f[['Player','Squad','League','Age','Similarity_Score','Market_Value_M','VES','Arbitrage_M','Decision_Score','Recommendation']].copy()
        disp.columns=['Player','Club','League','Age','Similarity','Value (€M)','VES','Saving (€M)','Decision','Rec']
        for c in ['Similarity','Decision','VES']: disp[c]=disp[c].map('{:.4f}'.format)
        st.dataframe(disp, use_container_width=True, hide_index=True)

# ════════════════ PLAYER PROFILE ════════════════
elif page == "👤 Player Profile":
    st.markdown("<h1>PLAYER PROFILE</h1>", unsafe_allow_html=True)
    bench_sel = st.selectbox("Benchmark", ["Mo Salah","Kevin De Bruyne"])
    bench_df  = master[master['Benchmark']==bench_sel].sort_values('Decision_Score',ascending=False)
    p_sel     = st.selectbox("Select Player", bench_df['Player'].tolist())
    row       = bench_df[bench_df['Player']==p_sel].iloc[0]
    bench_v   = salah_bench if bench_sel=='Mo Salah' else kdb_bench
    bench_c   = '#E63946' if bench_sel=='Mo Salah' else '#457B9D'

    st.markdown(f"""<div class='player-card' style='border-color:{bench_c};'>
        <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
            <div>
                <div style='font-family:Bebas Neue;font-size:2.2rem;'>{row['Player']}</div>
                <div style='color:#8B9DC3;'>{row['Squad']} · {row['League']} · Age {int(row['Age'])} · {row['Pos']}</div>
            </div>
            <div style='text-align:right;'>{badge_html(row['Recommendation'])}
                <div style='font-family:Bebas Neue;font-size:2rem;color:{bench_c};margin-top:4px;'>{row['Similarity_Score']:.1%}</div>
                <div style='color:#8B9DC3;font-size:0.75rem;'>vs {bench_sel}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Market Value",   f"€{row['Market_Value_M']:.1f}M")
    k2.metric("Arbitrage Save", f"€{row['Arbitrage_M']:.1f}M")
    k3.metric("VES",            f"{row['VES']:.4f}")
    k4.metric("Decision Score", f"{row['Decision_Score']:.4f}")
    k5.metric("Goals/90",       f"{row['Gls_90']:.3f}")
    st.divider()

    col1, col2 = st.columns(2)
    player_vals = {f: row[f] for f in features}
    with col1:
        st.markdown("<h3>Statistical Profile vs Benchmark</h3>", unsafe_allow_html=True)
        st.plotly_chart(make_radar(player_vals, row['Player'], bench_v, bench_sel, bench_c), use_container_width=True)
    with col2:
        st.markdown("<h3>Feature Strength (% of Benchmark)</h3>", unsafe_allow_html=True)
        fd = [{'Feature':f, 'Pct': min(row[f]/bench_v[f]*100,150) if bench_v[f]>0 else 0} for f in features]
        fd_df = pd.DataFrame(fd)
        fig_f = go.Figure()
        fig_f.add_trace(go.Bar(x=fd_df['Pct'], y=fd_df['Feature'], orientation='h',
            marker_color=['#2A9D8F' if p>=70 else '#E63946' if p<50 else '#E9C46A' for p in fd_df['Pct']],
            text=fd_df['Pct'].apply(lambda x: f'{x:.0f}%'), textposition='outside'))
        fig_f.add_vline(x=70,  line_dash='dash', line_color='#4A5568', annotation_text='70% min')
        fig_f.add_vline(x=100, line_dash='dot',  line_color='#8B9DC3', annotation_text='Benchmark')
        fig_f.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title='% of Benchmark', xaxis=dict(range=[0,160],gridcolor='#1E2D40'))
        st.plotly_chart(fig_f, use_container_width=True)

    st.divider()
    st.markdown("<h3>ScoutIQ Recommendation Rationale</h3>", unsafe_allow_html=True)
    strengths = [lbl for f,lbl in [('Gls_90','goal output'),('Ast_90','assist creation'),('Sh_90','shot volume'),('SoT_90','shooting accuracy'),('SoT_pct','shot efficiency'),('G_per_Sh','finishing quality')] if bench_v[f]>0 and row[f]>=bench_v[f]*0.70]
    st.info(f"**{row['Player']}** (Age {int(row['Age'])}, {row['Squad']}, {row['League']}) demonstrates **{row['Similarity_Score']:.1%} statistical similarity** to {bench_sel}. Key strengths: {', '.join(strengths) if strengths else 'developing profile'}. At €{row['Market_Value_M']:.1f}M market value, this represents a **€{row['Arbitrage_M']:.1f}M transfer saving** vs a statistically equivalent European player. Decision Score: **{row['Decision_Score']:.4f}**.")

# ════════════════ MARKET INTELLIGENCE ════════════════
elif page == "💹 Market Intelligence":
    st.markdown("<h1>MARKET INTELLIGENCE</h1>", unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["📈 Arbitrage","💰 Value Efficiency","🔮 Efficiency Frontier"])

    with tab1:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Salah Savings", f"€{master[master['Benchmark']=='Mo Salah']['Arbitrage_M'].sum():.0f}M")
        c2.metric("Total KDB Savings",   f"€{master[master['Benchmark']=='Kevin De Bruyne']['Arbitrage_M'].sum():.0f}M")
        c3.metric("Max Single Saving",   f"€{master['Arbitrage_M'].max():.1f}M")
        c4.metric("Avg Saving/Player",   f"€{master['Arbitrage_M'].mean():.1f}M")
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("<h3>Salah Targets</h3>", unsafe_allow_html=True)
            s_a = master[(master['Benchmark']=='Mo Salah')&(master['Arbitrage_M']>0)].nlargest(10,'Arbitrage_M')
            fig_a1 = px.bar(s_a,x='Player',y='Arbitrage_M',color='League',color_discrete_map=colors,text='Arbitrage_M',hover_data=['Market_Value_M','Similarity_Score'])
            fig_a1.update_traces(texttemplate='€%{text:.1f}M',textposition='outside')
            fig_a1.add_hline(y=15,line_dash='dash',line_color='#E63946',annotation_text='€15M Benchmark')
            fig_a1.update_layout(**PLOTLY_LAYOUT,height=400,xaxis_tickangle=-30,yaxis_title='Transfer Saving (€M)',xaxis=dict(gridcolor='#1E2D40'))
            st.plotly_chart(fig_a1,use_container_width=True)
        with col2:
            st.markdown("<h3>KDB Targets</h3>", unsafe_allow_html=True)
            k_a = master[(master['Benchmark']=='Kevin De Bruyne')&(master['Arbitrage_M']>0)].nlargest(10,'Arbitrage_M')
            fig_a2 = px.bar(k_a,x='Player',y='Arbitrage_M',color='League',color_discrete_map=colors,text='Arbitrage_M',hover_data=['Market_Value_M','Similarity_Score'])
            fig_a2.update_traces(texttemplate='€%{text:.1f}M',textposition='outside')
            fig_a2.add_hline(y=18,line_dash='dash',line_color='#457B9D',annotation_text='€18M Benchmark')
            fig_a2.update_layout(**PLOTLY_LAYOUT,height=400,xaxis_tickangle=-30,yaxis_title='Transfer Saving (€M)',xaxis=dict(gridcolor='#1E2D40'))
            st.plotly_chart(fig_a2,use_container_width=True)

    with tab2:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("<h3>VES — Salah Profile</h3>", unsafe_allow_html=True)
            s_v = master[master['Benchmark']=='Mo Salah'].nlargest(10,'VES')
            fig_v1 = px.bar(s_v,x='VES',y='Player',color='League',color_discrete_map=colors,orientation='h',text='VES',hover_data=['Market_Value_M','Similarity_Score'])
            fig_v1.update_traces(texttemplate='%{text:.4f}',textposition='outside')
            fig_v1.update_layout(**PLOTLY_LAYOUT,height=400,yaxis=dict(autorange='reversed'),xaxis_title='VES')
            st.plotly_chart(fig_v1,use_container_width=True)
        with col2:
            st.markdown("<h3>VES — KDB Profile</h3>", unsafe_allow_html=True)
            k_v = master[master['Benchmark']=='Kevin De Bruyne'].nlargest(10,'VES')
            fig_v2 = px.bar(k_v,x='VES',y='Player',color='League',color_discrete_map=colors,orientation='h',text='VES',hover_data=['Market_Value_M','Similarity_Score'])
            fig_v2.update_traces(texttemplate='%{text:.4f}',textposition='outside')
            fig_v2.update_layout(**PLOTLY_LAYOUT,height=400,yaxis=dict(autorange='reversed'),xaxis_title='VES')
            st.plotly_chart(fig_v2,use_container_width=True)

    with tab3:
        st.markdown("<h3>Efficiency Frontier — Ideal players: top-left corner</h3>", unsafe_allow_html=True)
        fig_bub = px.scatter(master,x='Market_Value_M',y='Similarity_Score',color='League',symbol='Benchmark',size='VES',size_max=28,hover_data=['Player','Squad','Age','Recommendation','Decision_Score'],color_discrete_map=colors,opacity=0.85)
        fig_bub.add_vline(x=15,line_dash='dash',line_color='#E63946',annotation_text='Salah Roma €15M')
        fig_bub.add_vline(x=18,line_dash='dash',line_color='#457B9D',annotation_text='KDB Wolfsburg €18M')
        fig_bub.add_annotation(x=1.5,y=0.72,text="🎯 IDEAL ZONE",showarrow=False,font=dict(size=12,color='#2A9D8F'),bgcolor='#2A9D8F22',bordercolor='#2A9D8F')
        fig_bub.update_layout(**PLOTLY_LAYOUT,height=550,xaxis_title='Market Value (€M)',yaxis_title='Similarity Score',xaxis=dict(gridcolor='#1E2D40'),yaxis=dict(gridcolor='#1E2D40'))
        st.plotly_chart(fig_bub,use_container_width=True)

# ════════════════ ABOUT ════════════════
elif page == "📦 About":
    st.markdown("<h1>ABOUT SCOUTIQ</h1>", unsafe_allow_html=True)
    col1,col2 = st.columns([2,1])
    with col1:
        st.markdown("""<div class='player-card'>
            <h3>What is ScoutIQ?</h3>
            <p style='color:#8B9DC3;line-height:1.8;'>ScoutIQ is a B2B sports analytics platform that uses machine learning to identify statistically undervalued football talent in European leagues. The system compares 264 U23 players from the Eredivisie, Primeira Liga, and EFL Championship against elite benchmark profiles using a hybrid similarity model (60% Cosine + 40% Euclidean), extended with a financial decision framework that addresses four documented research gaps in sports analytics literature.</p>
            <h3>The Business Case</h3>
            <p style='color:#8B9DC3;line-height:1.8;'>Liverpool bought Mo Salah for £36.9M in 2017. Roma bought him for €15M in 2016 — with identical statistics. ScoutIQ identifies players at the €15M moment, not the £36.9M moment. The model has identified transfer market savings of up to <b style='color:#F4A261;'>€14.2M per signing</b>.</p>
            <h3>Methodology</h3>
            <p style='color:#8B9DC3;line-height:1.8;'>Built on the CRISP-DM framework across 7 phases. ML pipeline: StandardScaler → PCA (77.6% variance explained) → K-Means K=4 → Hybrid Similarity Scoring → Financial Decision Framework. Data: FBref.com (Season 2023/24) + Transfermarkt valuations (Aug/Sep 2024).</p>
            <h3>Research Gaps Addressed</h3>
            <p style='color:#8B9DC3;line-height:1.8;'>
            ✅ Gap 1 — Prediction → Decision (Final Decision Score: 50% Sim + 30% VES + 20% Perf)<br>
            ✅ Gap 2 — Explainability (Feature heatmap + natural language explanations)<br>
            ✅ Gap 3 — Financial Integration (VES + Arbitrage + Cost per Goal)<br>
            ✅ Gap 4 — Decision Rules (STRONG BUY / BUY / MONITOR / PASS engine)</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='player-card' style='border-color:#F4A261;'>
            <div style='font-family:Bebas Neue;font-size:1.5rem;color:#F4A261;'>PRICING</div><hr style='border-color:#1E2D40;'>
            <div style='margin-bottom:1rem;'><div style='font-weight:700;'>Starter</div>
                <div style='font-size:1.5rem;font-weight:700;color:#2A9D8F;'>€299<span style='font-size:0.9rem;color:#8B9DC3;'>/mo</span></div>
                <div style='color:#8B9DC3;font-size:0.8rem;'>Egyptian & Moroccan clubs</div></div>
            <div style='margin-bottom:1rem;'><div style='font-weight:700;'>Professional</div>
                <div style='font-size:1.5rem;font-weight:700;color:#E9C46A;'>€999<span style='font-size:0.9rem;color:#8B9DC3;'>/mo</span></div>
                <div style='color:#8B9DC3;font-size:0.8rem;'>Saudi Pro League clubs</div></div>
            <div><div style='font-weight:700;'>Enterprise</div>
                <div style='font-size:1.5rem;font-weight:700;color:#E63946;'>€3,999<span style='font-size:0.9rem;color:#8B9DC3;'>/mo</span></div>
                <div style='color:#8B9DC3;font-size:0.8rem;'>European scouting depts</div></div>
        </div>
        <div class='player-card' style='margin-top:1rem;'>
            <div style='font-family:Bebas Neue;font-size:1.2rem;color:#8B9DC3;'>PROJECT INFO</div><hr style='border-color:#1E2D40;'>
            <div style='color:#8B9DC3;font-size:0.85rem;line-height:2;'>👤 Abdelrahman M. Elhosary<br>🎓 USM · ABW508 Analytics Lab<br>👨‍🏫 Dr. Muhammad Shabir Bin Shaharudin<br>📊 264 players · 3 leagues · 34 valued<br>🤖 PCA 77.6% · K-Means K=4 · Hybrid Sim<br>💶 Max saving: €14.2M per signing<br>🏆 7 CRISP-DM phases · 4 gaps solved</div>
        </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='text-align:center;color:#4A5568;font-size:0.8rem;'>ScoutIQ · Master of Business Analytics · Universiti Sains Malaysia · ABW508 Analytics Lab · 2024/25</div>", unsafe_allow_html=True)
