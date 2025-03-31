import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob
import os
import base64
from datetime import datetime

# Trackmanデータ読み込み関数
@st.cache_data(ttl=3600)
def load_trackman_data():
    """
    data_trackmanディレクトリからCSVファイルを読み込み、結合する関数
    """
    # 現在のディレクトリ確認
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'data_trackman')
    
    if not os.path.exists(data_dir):
        st.error(f"data_trackmanディレクトリが見つかりません: {data_dir}")
        return None
    
    # 再帰的にCSVファイルを検索
    trackman_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    
    if not trackman_files:
        st.error(f"data_trackmanディレクトリにCSVファイルが見つかりません: {data_dir}")
        return None
    
    # 進捗表示用
    progress_bar = st.progress(0)
    dfs = []
    
    for i, file in enumerate(trackman_files):
        try:
            df = pd.read_csv(file, encoding='utf-8-sig', low_memory=False)
            # URLカラムがない場合は追加
            if 'c_URL' not in df.columns:
                df['c_URL'] = ''
            dfs.append(df)
        except Exception as e:
            st.warning(f"ファイル '{os.path.basename(file)}' の読み込み中にエラー: {e}")
            continue
        
        # 進捗更新
        progress_bar.progress((i + 1) / len(trackman_files))
    
    progress_bar.empty()  # 進捗バーを消去
    
    if not dfs:
        st.error("データを読み込めませんでした。")
        return None
    
    # データフレームを結合
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 必要なカラムが含まれているか確認
    required_columns = ['Pitcher', 'Batter', 'BatterSide', 'TaggedPitchType', 'PlateLocSide', 'PlateLocHeight']
    missing_columns = [col for col in required_columns if col not in combined_df.columns]
    
    if missing_columns:
        st.warning(f"データに必要なカラムが不足しています: {', '.join(missing_columns)}")
    
    # データ前処理
    combined_df = preprocess_trackman_data(combined_df)
    
    return combined_df

def preprocess_trackman_data(df):
    """
    Trackmanデータの前処理を行う関数
    """
    # NaNを処理
    df = df.fillna({
        'TaggedPitchType': 'Undefined',
        'PlayResult': 'Undefined',
        'PitchCall': 'Undefined'
    })
    
    # プロットに必要なカラムを確認
    if 'PlateLocSide' in df.columns and 'PlateLocHeight' in df.columns:
        # 値の範囲チェック
        df = df[(-5 < df['PlateLocSide']) & (df['PlateLocSide'] < 5)]
        df = df[(0 < df['PlateLocHeight']) & (df['PlateLocHeight'] < 5)]
    
    # モード確認 (opensen/league)
    # ここでdata_trackmanのデータが100倍されている場合はopsensen、そうでない場合はleagueモード
    if 'PlateLocSide' in df.columns:
        mean_abs_side = df['PlateLocSide'].abs().mean()
        if mean_abs_side > 5:  # 値が大きすぎる場合はopensen形式と判断
            df['PlateLocSide'] = df['PlateLocSide'] * 0.01
            df['PlateLocHeight'] = df['PlateLocHeight'] * 0.01
    
    return df

# コースプロット関数 (BTZplot_plotly)
def BTZplot_plotly(df, df_left=None, df_right=None, name='default', title='コースプロット', mode='league', 
                   ss=10, alpha=0.5, legend=False, txt=False, hosei=True, C=False, rtnfig=True, url=True):
    if df.empty:
        return 'No data'
    df1 = df.dropna(subset=['TaggedPitchType','PlateLocSide','PlateLocHeight'])
    if mode == 'opensen':
        df1['PlateLocSide'] = df1['PlateLocSide'] * 0.01
        df1['PlateLocHeight'] = df1['PlateLocHeight'] * 0.01
    if df_left is None and df_right is None:
        df_left = df1[df1['BatterSide'] == 'Left']
        df_right = df1[df1['BatterSide'] == 'Right']
    elif df_left is None:
        df_left = pd.DataFrame()  # 空のDataFrameを作成
    elif df_right is None:
        df_right = pd.DataFrame()  # 空のDataFrameを作成
    
    if name != 'default':
        df1 = df1[df1['Pitcher'] == name]
        df_left = df_left[df_left['Pitcher'] == name]
        df_right = df_right[df_right['Pitcher'] == name]

    pitch_types = ['Fastball', 'Slider', 'Cutter', 'Curveball', 'ChangeUp', 'Splitter', 'Sinker', 'Knuckleball', 'TwoSeamFastBall', 'Other', 'Undefined']
    colors = ['red', 'limegreen', 'darkviolet', 'orange', 'deepskyblue', 'magenta', 'blue', 'silver', 'dodgerblue', 'black', 'grey']
    marks = [['circle', 'triangle-left', 'triangle-left', 'triangle-up', 'triangle-down', 'triangle-down', 'triangle-right', 'star', 'square', 'cross', 'x'], ['circle', 'triangle-right', 'diamond', 'triangle-up', 'diamond-wide', 'triangle-down', 'hexagon', 'star', 'hexagon2', 'cross', 'x']]

    fig = go.Figure()

    for pt, color, mark_left, mark_right in zip(pitch_types, colors, marks[0], marks[1]):
        for side, df_side in [('Left', df_left), ('Right', df_right)]:
            if not df_side.empty:
                df_pt = df_side[df_side['TaggedPitchType'] == pt]
                if not df_pt.empty:
                    fig.add_trace(go.Scatter(
                        x=df_pt['PlateLocSide'], 
                        y=df_pt['PlateLocHeight'], 
                        mode='markers+text' if txt else 'markers',
                        marker=dict(color=color, size=ss, opacity=alpha, symbol=mark_left if side == 'Left' else mark_right),
                        name=f"{pt} ({side})",
                        text=df_pt['PitchofPA'].apply(lambda x: int(round(x, 0)) if pd.notna(x) else '') if txt else None,
                        textposition='top center',
                        customdata=df_pt['c_URL'].tolist() if url else None
                    ))

    fig.update_layout(
        title={'text': title, 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis=dict(range=[-0.9, 0.9], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0.0, 1.8], showgrid=False, zeroline=False, scaleanchor='x'),
        width=650,
        height=650,
        showlegend=legend,
    )

    # ストライクゾーンの実線
    strike_zone_shapes = [
        dict(type="line", x0=-0.30, y0=0.4495, x1=-0.30, y1=1.0505, line=dict(color="black", width=1)),
        dict(type="line", x0=0.30, y0=0.4495, x1=0.30, y1=1.0505, line=dict(color="black", width=1)),
        dict(type="line", x0=-0.30, y0=0.45, x1=0.30, y1=0.45, line=dict(color="black", width=1)),
        dict(type="line", x0=-0.30, y0=1.05, x1=0.30, y1=1.05, line=dict(color="black", width=1)),
        dict(type="line", x0=-0.10, y0=0.45, x1=-0.10, y1=1.05, line=dict(color="black", width=1, dash="dash")),
        dict(type="line", x0=0.10, y0=0.45, x1=0.10, y1=1.05, line=dict(color="black", width=1, dash="dash")),
        dict(type="line", x0=-0.30, y0=0.65, x1=0.30, y1=0.65, line=dict(color="black", width=1, dash="dash")),
        dict(type="line", x0=-0.30, y0=0.85, x1=0.30, y1=0.85, line=dict(color="black", width=1, dash="dash"))
    ]
    for shape in strike_zone_shapes:
        fig.add_shape(**shape)

    # updatemenus の設定
    if df_left.empty or df_right.empty:
        updatemenus = []  # 左または右のデータフレームが空の場合、updatemenus を空に
    else:
        updatemenus = [
            dict(
                type="dropdown",
                direction="down",
                x=0.1,
                y=1.1,
                showactive=True,
                buttons=list([
                    dict(label="Both", method="update", args=[{"visible": [True, True]}, {"title": "BatterSide: Both"}]),
                    dict(label="Left", method="update", args=[{"visible": [True, False]}, {"title": "BatterSide: Left"}]),
                    dict(label="Right", method="update", args=[{"visible": [False, True]}, {"title": "BatterSide: Right"}])
                ]),
            )
        ]
    fig.update_layout(updatemenus=updatemenus)

    if rtnfig:
        return fig
    else:
        if url:
            html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
            html_str += """
            <script>
            document.querySelectorAll('.plotly-graph-div').forEach(function(plotDiv) {
                plotDiv.on('plotly_click', function(data){
                    var point = data.points[0];
                    if(point && point.customdata) {
                        window.open(point.customdata, '_blank');
                    }
                });
            });
            </script>
            """

            return html_str

# CSV出力モード用のインタラクティブプロット関数
def create_interactive_plot_for_csv(df, df_left, df_right, title):
    """
    CSV出力モード用のインタラクティブプロットを作成する
    クリックされた点は、選択済みとして色が変わり、セッションに保存される
    """
    # BTZplot_plotlyのクローンだが、インタラクティブ選択機能を追加
    pitch_types = ['Fastball', 'Slider', 'Cutter', 'Curveball', 'ChangeUp', 'Splitter', 'Sinker', 'Knuckleball', 'TwoSeamFastBall', 'Other', 'Undefined']
    colors = ['red', 'limegreen', 'darkviolet', 'orange', 'deepskyblue', 'magenta', 'blue', 'silver', 'dodgerblue', 'black', 'grey']
    marks = [['circle', 'triangle-left', 'triangle-left', 'triangle-up', 'triangle-down', 'triangle-down', 'triangle-right', 'star', 'square', 'cross', 'x'], 
             ['circle', 'triangle-right', 'diamond', 'triangle-up', 'diamond-wide', 'triangle-down', 'hexagon', 'star', 'hexagon2', 'cross', 'x']]
    
    fig = go.Figure()
    
    # すでに選択済みの投球インデックスリスト
    selected_indices = st.session_state.selected_pitches
    
    for pt, color, mark_left, mark_right in zip(pitch_types, colors, marks[0], marks[1]):
        for side, df_side in [('Left', df_left), ('Right', df_right)]:
            if not df_side.empty:
                df_pt = df_side[df_side['TaggedPitchType'] == pt]
                
                if not df_pt.empty:
                    # 未選択の投球（薄い色）
                    unselected = df_pt[~df_pt.index.isin(selected_indices)]
                    if not unselected.empty:
                        fig.add_trace(go.Scatter(
                            x=unselected['PlateLocSide'], 
                            y=unselected['PlateLocHeight'], 
                            mode='markers',
                            marker=dict(color=color, size=10, opacity=0.3, symbol=mark_left if side == 'Left' else mark_right),
                            name=f"{pt} ({side}) - 未選択",
                            customdata=unselected.index.tolist(),
                            hovertemplate="投球: %{customdata}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"
                        ))
                    
                    # 選択済みの投球（濃い色）
                    selected = df_pt[df_pt.index.isin(selected_indices)]
                    if not selected.empty:
                        fig.add_trace(go.Scatter(
                            x=selected['PlateLocSide'], 
                            y=selected['PlateLocHeight'], 
                            mode='markers',
                            marker=dict(color=color, size=10, opacity=1.0, symbol=mark_left if side == 'Left' else mark_right),
                            name=f"{pt} ({side}) - 選択済み",
                            customdata=selected.index.tolist(),
                            hovertemplate="投球: %{customdata}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"
                        ))
    
    # ストライクゾーンの描画
    strike_zone_shapes = [
        dict(type="line", x0=-0.30, y0=0.4495, x1=-0.30, y1=1.0505, line=dict(color="black", width=1)),
        dict(type="line", x0=0.30, y0=0.4495, x1=0.30, y1=1.0505, line=dict(color="black", width=1)),
        dict(type="line", x0=-0.30, y0=0.45, x1=0.30, y1=0.45, line=dict(color="black", width=1)),
        dict(type="line", x0=-0.30, y0=1.05, x1=0.30, y1=1.05, line=dict(color="black", width=1)),
        dict(type="line", x0=-0.10, y0=0.45, x1=-0.10, y1=1.05, line=dict(color="black", width=1, dash="dash")),
        dict(type="line", x0=0.10, y0=0.45, x1=0.10, y1=1.05, line=dict(color="black", width=1, dash="dash")),
        dict(type="line", x0=-0.30, y0=0.65, x1=0.30, y1=0.65, line=dict(color="black", width=1, dash="dash")),
        dict(type="line", x0=-0.30, y0=0.85, x1=0.30, y1=0.85, line=dict(color="black", width=1, dash="dash"))
    ]
    for shape in strike_zone_shapes:
        fig.add_shape(**shape)
    
    # プロットレイアウト
    fig.update_layout(
        title={'text': title, 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis=dict(range=[-0.9, 0.9], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0.0, 1.8], showgrid=False, zeroline=False, scaleanchor='x'),
        width=650,
        height=650,
        showlegend=False,
    )
    
    # クリックイベントのコールバック
    fig.update_layout(clickmode='event')
    
    return fig

# コースプロットページの関数
def pitch_plot_page():
    """
    コースプロット表示ページの関数
    """
    st.title("投球コースプロット")
    
    with st.expander("使い方"):
        st.write("1. 左のサイドバーで、投手名・打者名・カウント・結果などでフィルタリングができます。")
        st.write("2. プロット上のポイントをクリックすると、関連する動画のリンクに遷移します。")
        st.write("3. 「CSV出力モード」を有効にすると、プロット上のポイントをクリックして選択できます。")
        st.write("4. 選択したポイントは色が濃くなり、「CSV出力」ボタンでその投球データをダウンロードできます。")
    
    # サイドバーにプロットボタンを追加
    # 検索条件入力UIを表示
    st.sidebar.subheader("フィルタ設定")
    
    # データロードはフィルタ条件を設定した後で行う（遅延ロード）
    with st.spinner("Trackmanデータを読み込んでいます..."):
        df_trackman = load_trackman_data()
    
    if df_trackman is None or df_trackman.empty:
        st.error("データが取得できませんでした。")
        return
    
    # カウント選択
    pitcher_options = sorted(df_trackman["Pitcher"].dropna().unique().astype(str).tolist()) if "Pitcher" in df_trackman.columns else []
    batter_options = sorted(df_trackman["Batter"].dropna().unique().astype(str).tolist()) if "Batter" in df_trackman.columns else []
    balls_options = sorted(df_trackman["Balls"].dropna().unique().astype(int).tolist()) if "Balls" in df_trackman.columns else []
    strikes_options = sorted(df_trackman["Strikes"].dropna().unique().astype(int).tolist()) if "Strikes" in df_trackman.columns else []
    outs_options = sorted(df_trackman["Outs"].dropna().unique().astype(int).tolist()) if "Outs" in df_trackman.columns else []
    
    pitcher = st.sidebar.multiselect("投手名", pitcher_options)
    batter = st.sidebar.multiselect("打者名", batter_options)
    balls = st.sidebar.multiselect("ボール数", balls_options)
    strikes = st.sidebar.multiselect("ストライク数", strikes_options)
    outs = st.sidebar.multiselect("アウト数", outs_options)
    
    # 結果選択
    pitch_call_options = sorted(df_trackman["PitchCall"].dropna().unique().tolist()) if "PitchCall" in df_trackman.columns else []
    play_result_options = sorted(df_trackman["PlayResult"].dropna().unique().tolist()) if "PlayResult" in df_trackman.columns else []
    
    pitch_calls = st.sidebar.multiselect("PitchCall", pitch_call_options)
    play_results = st.sidebar.multiselect("PlayResult", play_result_options)
    
    # CSV出力モード
    csv_mode = st.sidebar.checkbox("CSV出力モード")
    
    # セッション状態の初期化
    if 'selected_pitches' not in st.session_state:
        st.session_state.selected_pitches = []
    
    if 'last_clicked_index' not in st.session_state:
        st.session_state.last_clicked_index = None
    
    # プロットボタン
    plot_button = st.sidebar.button("プロット表示")
    
    # クエリパラメータからの選択状態の更新処理
    query_params = st.query_params
    if "selected" in query_params:
        try:
            index = int(query_params["selected"])
            # トグル動作: すでに選択されていれば削除、なければ追加
            if index in st.session_state.selected_pitches:
                st.session_state.selected_pitches.remove(index)
            else:
                st.session_state.selected_pitches.append(index)
            st.query_params.clear()  # クエリパラメータをクリア
            st.rerun()  # ページを再読み込み
        except ValueError:
            pass
    
    # プロット表示ボタンが押された、またはフィルタ条件が指定されていない場合
    if plot_button or (not pitcher and not batter and not balls and not strikes and not outs and not pitch_calls and not play_results):
        # フィルタリング適用
        filtered_df = df_trackman.copy()
        
        if pitcher:
            filtered_df = filtered_df[filtered_df["Pitcher"].isin(pitcher)]
        if batter:
            filtered_df = filtered_df[filtered_df["Batter"].isin(batter)]
        if balls:
            filtered_df = filtered_df[filtered_df["Balls"].isin(balls)]
        if strikes:
            filtered_df = filtered_df[filtered_df["Strikes"].isin(strikes)]
        if outs:
            filtered_df = filtered_df[filtered_df["Outs"].isin(outs)]
        if pitch_calls:
            filtered_df = filtered_df[filtered_df["PitchCall"].isin(pitch_calls)]
        if play_results:
            filtered_df = filtered_df[filtered_df["PlayResult"].isin(play_results)]
        
        # メインコンテンツ
        if not filtered_df.empty:
            st.write(f"表示データ数: {len(filtered_df)}件")
            
            # 左打者と右打者でデータを分ける
            df_left = filtered_df[filtered_df['BatterSide'] == 'Left']
            df_right = filtered_df[filtered_df['BatterSide'] == 'Right']
            
            # プロットタイトル
            plot_title = "コースプロット"
            if pitcher:
                plot_title += f" 投手: {', '.join(pitcher)}"
            if batter:
                plot_title += f" 打者: {', '.join(batter)}"
            
            # CSV出力モードに応じたプロット設定
            if csv_mode:
                # CSV出力モード用のカスタム設定
                fig = create_interactive_plot_for_csv(filtered_df, df_left, df_right, plot_title)
                
                # クリックイベント処理用のJavaScript
                st.markdown(
                    """
                    <script>
                    const graphDiv = document.querySelector('.js-plotly-plot');
                    if (graphDiv) {
                        graphDiv.on('plotly_click', function(data) {
                            if (data.points.length > 0) {
                                const point = data.points[0];
                                const index = point.customdata;
                                if (index !== undefined) {
                                    // URLパラメータを更新して自動的にページをリロード
                                    const url = new URL(window.location);
                                    url.searchParams.set('selected', index);
                                    window.history.pushState({}, '', url);
                                    window.location.href = url;
                                }
                            }
                        });
                    }
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                
                # 選択状態表示
                st.write(f"選択済み投球数: {len(st.session_state.selected_pitches)}")
                
                # クリア/保存ボタン
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("選択をクリア"):
                        st.session_state.selected_pitches = []
                        st.rerun()
                
                with col2:
                    if st.button("CSV出力"):
                        if st.session_state.selected_pitches:
                            # インデックスで選択された投球を抽出
                            selected_df = filtered_df.loc[st.session_state.selected_pitches]
                            
                            # CSVダウンロード
                            csv = selected_df.to_csv(index=False).encode('utf-8-sig')
                            b64 = base64.b64encode(csv).decode()
                            
                            # 現在時刻を含めたファイル名
                            now = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"selected_pitches_{now}.csv"
                            
                            # ダウンロードリンク
                            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">CSVファイルをダウンロード</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        else:
                            st.warning("投球が選択されていません。プロット上の点をクリックして選択してください。")
            else:
                # 通常モード：URLリンク付きプロット
                fig = BTZplot_plotly(filtered_df, df_left, df_right, 
                                   title=plot_title,
                                   rtnfig=True,
                                   url=True)
                
                # URLクリック用のJavaScript
                st.markdown(
                    """
                    <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        const plots = document.querySelectorAll('.plotly-graph-div');
                        plots.forEach(function(plot) {
                            plot.on('plotly_click', function(data) {
                                const point = data.points[0];
                                if (point && point.customdata) {
                                    window.open(point.customdata, '_blank');
                                }
                            });
                        });
                    });
                    </script>
                    """,
                    unsafe_allow_html=True
                )
            
            # プロットの表示
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("フィルタリング条件に一致するデータがありません。フィルタを調整してください。")
    else:
        st.info("フィルタ条件を入力して「プロット表示」ボタンをクリックしてください。")
