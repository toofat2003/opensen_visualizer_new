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
    # 注意: 'data_trackman' ディレクトリがスクリプトと同じ階層にあることを想定
    data_dir = os.path.join(cwd, 'data_trackman')

    if not os.path.exists(data_dir):
        st.error(f"data_trackmanディレクトリが見つかりません: {data_dir}")
        st.info(f"現在の作業ディレクトリ: {cwd}")
        # 一般的な場所も探してみる (例: 親ディレクトリの data_trackman)
        parent_data_dir = os.path.join(os.path.dirname(cwd), 'data_trackman')
        if os.path.exists(parent_data_dir):
            data_dir = parent_data_dir
            st.info(f"親ディレクトリで発見: {data_dir}")
        else:
             # さらに親も探す (デプロイ環境などでネストが深い場合)
            grandparent_data_dir = os.path.join(os.path.dirname(os.path.dirname(cwd)), 'data_trackman')
            if os.path.exists(grandparent_data_dir):
                data_dir = grandparent_data_dir
                st.info(f"2階層上の親ディレクトリで発見: {data_dir}")
            else:
                st.error(f"一般的な場所でも data_trackman が見つかりませんでした。")
                return None # ここでNoneを返す

    # 再帰的にCSVファイルを検索
    trackman_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)

    if not trackman_files:
        st.error(f"data_trackmanディレクトリ ({data_dir}) にCSVファイルが見つかりません。")
        return None # ここでNoneを返す

    # 進捗表示用
    progress_bar = st.progress(0)
    progress_text = st.empty()
    dfs = []

    total_files = len(trackman_files)
    for i, file in enumerate(trackman_files):
        filename = os.path.basename(file)
        progress_text.text(f"読み込み中 ({i+1}/{total_files}): {filename}")
        try:
            # encoding='utf-8-sig' でBOM付きUTF-8に対応
            df = pd.read_csv(file, encoding='utf-8-sig', low_memory=False)
            # URLカラムがない場合は空文字列で追加
            if 'c_URL' not in df.columns:
                df['c_URL'] = ''
            # 必須カラムがない場合も警告を出すが、処理は続ける
            required_cols = ['Pitcher', 'Batter', 'BatterSide', 'TaggedPitchType', 'PlateLocSide', 'PlateLocHeight', 'PlayResult', 'PitchCall', 'Balls', 'Strikes', 'Outs', 'PitchofPA']
            for col in required_cols:
                if col not in df.columns:
                    st.warning(f"ファイル '{filename}' に必須カラム '{col}' がありません。処理は続行しますが、予期せぬ動作の可能性があります。")
                    # 欠損カラムをNaNで埋めるか、デフォルト値で埋めるか検討 (ここではNaNのままにする)
                    # df[col] = np.nan # or df[col] = 'Undefined' etc.
            dfs.append(df)
        except UnicodeDecodeError:
             st.warning(f"ファイル '{filename}' の読み込み中にエンコーディングエラーが発生しました。Shift-JISで再試行します。")
             try:
                 df = pd.read_csv(file, encoding='shift-jis', low_memory=False)
                 if 'c_URL' not in df.columns:
                     df['c_URL'] = ''
                 required_cols = ['Pitcher', 'Batter', 'BatterSide', 'TaggedPitchType', 'PlateLocSide', 'PlateLocHeight', 'PlayResult', 'PitchCall', 'Balls', 'Strikes', 'Outs', 'PitchofPA']
                 for col in required_cols:
                     if col not in df.columns:
                         st.warning(f"ファイル '{filename}' に必須カラム '{col}' がありません。処理は続行しますが、予期せぬ動作の可能性があります。")
                 dfs.append(df)
             except Exception as e_sj:
                 st.error(f"ファイル '{filename}' のShift-JISでの読み込みにも失敗しました: {e_sj}")
                 continue # 次のファイルへ
        except Exception as e:
            st.error(f"ファイル '{filename}' の読み込み中に予期せぬエラーが発生しました: {e}")
            continue # 次のファイルへ

        # 進捗更新
        progress_bar.progress((i + 1) / total_files)

    progress_bar.empty()  # 進捗バーを消去
    progress_text.empty() # テキストを消去

    if not dfs:
        st.error("有効なデータを読み込めませんでした。CSVファイルの内容と形式を確認してください。")
        return None # ここでNoneを返す

    # データフレームを結合
    try:
        combined_df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"データフレームの結合中にエラーが発生しました: {e}")
        return None

    # --- データ型の確認と修正 ---
    # 数値であるべきカラムを確認
    numeric_cols = ['PlateLocSide', 'PlateLocHeight', 'Balls', 'Strikes', 'Outs', 'PitchofPA']
    for col in numeric_cols:
        if col in combined_df.columns:
            # errors='coerce' は数値に変換できない値をNaNにする
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            # NaNになった値があるか確認
            if combined_df[col].isnull().any():
                 st.warning(f"カラム '{col}' に数値に変換できない値が含まれていました。該当箇所はNaNとして扱われます。")
        else:
             st.warning(f"結合後のデータに必須の数値カラム '{col}' がありません。")
             combined_df[col] = np.nan # カラム自体が存在しない場合はNaNで作成

    # 文字列であるべきカラムを確認 (NaN以外)
    string_cols = ['Pitcher', 'Batter', 'BatterSide', 'TaggedPitchType', 'PlayResult', 'PitchCall', 'c_URL']
    for col in string_cols:
        if col in combined_df.columns:
            # fillna('') でNaNを空文字列に変換してから文字列型へ
            combined_df[col] = combined_df[col].fillna('').astype(str)
        else:
             st.warning(f"結合後のデータに必須の文字列カラム '{col}' がありません。")
             combined_df[col] = '' # カラム自体が存在しない場合は空文字列で作成

    # データ前処理
    try:
        combined_df = preprocess_trackman_data(combined_df)
    except Exception as e:
        st.error(f"データ前処理中にエラーが発生しました: {e}")
        return None

    st.success(f"データの読み込みと前処理が完了しました。合計 {len(combined_df)} 件")
    return combined_df


def preprocess_trackman_data(df):
    """
    Trackmanデータの前処理を行う関数
    ★ PlateLocSide/Height が メートル(m)単位であることを前提に修正 ★
    """
    # --- NaN処理 ---
    # (変更なし)
    categorical_cols = ['TaggedPitchType', 'PlayResult', 'PitchCall', 'BatterSide']
    for col in categorical_cols:
        if col in df.columns: df[col] = df[col].fillna('Undefined')
        else: df[col] = 'Undefined'
    if 'c_URL' in df.columns: df['c_URL'] = df['c_URL'].fillna('')
    else: df['c_URL'] = ''

    # --- 単位変換と範囲チェック ---
    if 'PlateLocSide' in df.columns and 'PlateLocHeight' in df.columns:
        # 数値型に変換 (エラーはNaN)
        df['PlateLocSide'] = pd.to_numeric(df['PlateLocSide'], errors='coerce')
        df['PlateLocHeight'] = pd.to_numeric(df['PlateLocHeight'], errors='coerce')

        # --- スケーリング処理の削除 ---
        # 元データがメートル(m)単位であるため、mean_abs_side > 5 による
        # * 0.01 スケーリングは不要と判断し、削除またはコメントアウトします。
        # mean_abs_side = df['PlateLocSide'].dropna().abs().mean()
        # if pd.notna(mean_abs_side) and mean_abs_side > 5:
        #     st.info("座標値が大きいと判断しましたが、メートル単位を前提とするためスケーリングは行いません。")
            # df['PlateLocSide'] = df['PlateLocSide'] * 0.01 # 不要
            # df['PlateLocHeight'] = df['PlateLocHeight'] * 0.01 # 不要

        # 値の範囲外を除外 (メートル単位でのフィルタリング)
        # メートル単位として妥当な範囲に修正 (例: 左右±1m, 高さ0～2m)
        filter_range_side_lower = -1.0  # m
        filter_range_side_upper = 1.0   # m
        filter_range_height_lower = 0.0 # m
        filter_range_height_upper = 2.0 # m (少し余裕を持たせる)

        original_count = len(df)
        df = df[
            (filter_range_side_lower < df['PlateLocSide']) & (df['PlateLocSide'] < filter_range_side_upper) &
            (filter_range_height_lower < df['PlateLocHeight']) & (df['PlateLocHeight'] < filter_range_height_upper)
        ].copy()
        filtered_count = len(df)
        if original_count > filtered_count:
            st.info(f"座標(PlateLocSide/Height)が範囲外({filter_range_side_lower:.1f}m-{filter_range_side_upper:.1f}m / {filter_range_height_lower:.1f}m-{filter_range_height_upper:.1f}m)または無効な {original_count - filtered_count} 件のデータを除外しました。")

    else:
        st.warning("プロットに必要な 'PlateLocSide' または 'PlateLocHeight' カラムが見つかりません。コースプロットは表示できません。")
        if 'PlateLocSide' not in df.columns: df['PlateLocSide'] = np.nan
        if 'PlateLocHeight' not in df.columns: df['PlateLocHeight'] = np.nan

    # PitchofPAの整数化 (変更なし)
    if 'PitchofPA' in df.columns:
         df['PitchofPA'] = pd.to_numeric(df['PitchofPA'], errors='coerce')

    return df


# コースプロット関数 (BTZplot_plotly) - 通常モード用
def BTZplot_plotly(df, df_left=None, df_right=None, name='default', title='コースプロット',
                   mode='league', # mode引数はpreprocessで処理済みのため、本来不要かも
                   ss=10, alpha=0.5, legend=False, txt=False, hosei=True, C=False, rtnfig=True):
    """
    通常表示モード（クリックでURLを開く）のコースプロットを作成する関数
    """
    # 必須カラムの存在チェック
    required_plot_cols = ['TaggedPitchType','PlateLocSide','PlateLocHeight', 'BatterSide', 'Pitcher', 'Batter', 'PlayResult', 'c_URL']
    if df.empty or not all(col in df.columns for col in required_plot_cols):
        st.warning("プロットに必要なデータが不足しているため、表示できません。")
        # 空の Figure を返すか、エラーメッセージを含む Figure を返す
        fig = go.Figure()
        fig.update_layout(title=title + " (データ不足)", xaxis={'visible': False}, yaxis={'visible': False})
        return fig if rtnfig else "プロットに必要なデータが不足しています。"

    # NaNを除外 (特に座標)
    df1 = df.dropna(subset=['PlateLocSide','PlateLocHeight']).copy() # .copy()推奨
    # preprocessで実施済みだが念のため
    df1['TaggedPitchType'] = df1['TaggedPitchType'].fillna('Undefined')
    df1['c_URL'] = df1['c_URL'].fillna('')
    df1['PlayResult'] = df1['PlayResult'].fillna('Undefined')
    df1['Pitcher'] = df1['Pitcher'].fillna('Unknown')
    df1['Batter'] = df1['Batter'].fillna('Unknown')

    # モードに応じたスケーリング (preprocessで実施済みのはずなのでコメントアウトしても良い)
    # if mode == 'opensen':
    #     if 'PlateLocSide' in df1.columns: df1.loc[:, 'PlateLocSide'] = df1['PlateLocSide'] * 0.01
    #     if 'PlateLocHeight' in df1.columns: df1.loc[:, 'PlateLocHeight'] = df1['PlateLocHeight'] * 0.01

    # 左右打者データの準備
    if df_left is None and df_right is None:
        df_left = df1[df1['BatterSide'] == 'Left']
        df_right = df1[df1['BatterSide'] == 'Right']
    elif df_left is None:
        df_left = pd.DataFrame(columns=df1.columns) # 列を合わせて空のDF
    elif df_right is None:
        df_right = pd.DataFrame(columns=df1.columns) # 列を合わせて空のDF

    # 投手名フィルタ (現在は未使用？ name引数)
    if name != 'default':
        df_left = df_left[df_left['Pitcher'] == name]
        df_right = df_right[df_right['Pitcher'] == name]

    pitch_types = ['Fastball', 'Slider', 'Cutter', 'Curveball', 'ChangeUp', 'Splitter', 'Sinker', 'Knuckleball', 'TwoSeamFastBall', 'Other', 'Undefined']
    colors = ['red', 'limegreen', 'darkviolet', 'orange', 'deepskyblue', 'magenta', 'blue', 'silver', 'dodgerblue', 'black', 'grey']
    # 各球種に対応するマーカー（左打者用, 右打者用）
    marks = [['circle', 'triangle-left', 'triangle-left', 'triangle-up', 'triangle-down', 'triangle-down', 'triangle-right', 'star', 'square', 'cross', 'x'],
             ['circle', 'triangle-right', 'diamond', 'triangle-up', 'diamond-wide', 'triangle-down', 'hexagon', 'star', 'hexagon2', 'cross', 'x']]

    fig = go.Figure()

    # --- トレースの追加 ---
    visible_left = [] # 左打者トレースの可視性管理用
    visible_right = [] # 右打者トレースの可視性管理用

    for pt, color, mark_left, mark_right in zip(pitch_types, colors, marks[0], marks[1]):
        added_left = False
        added_right = False
        # 左打者データ
        if not df_left.empty:
            df_pt_left = df_left[df_left['TaggedPitchType'] == pt]
            if not df_pt_left.empty:
                # hover表示用に必要なデータをcustomdataに格納
                custom_data_left = df_pt_left[['Pitcher', 'Batter', 'PlayResult', 'c_URL']].fillna('N/A').values
                fig.add_trace(go.Scattergl(
                    x=df_pt_left['PlateLocSide'],
                    y=df_pt_left['PlateLocHeight'],
                    mode='markers+text' if txt else 'markers',
                    marker=dict(color=color, size=ss, opacity=alpha, symbol=mark_left),
                    name=f"{pt} (Left)", # 凡例名
                    text=df_pt_left['PitchofPA'].apply(lambda x: int(round(x, 0)) if pd.notna(x) else '') if txt else None,
                    textposition='top center',
                    customdata=custom_data_left, # ここに[Pitcher, Batter, PlayResult, c_URL]の配列を格納
                    hovertemplate=( # hoverテンプレートを定義
                        "<b>Pitcher:</b> %{customdata[0]}<br>"
                        "<b>Batter:</b> %{customdata[1]}<br>"
                        "<b>Result:</b> %{customdata[2]}<br>"
                        "<b>URL:</b> %{customdata[3]}<br>"
                        "X: %{x:.2f}, Y: %{y:.2f}<extra></extra>" # <extra>でデフォルト情報を非表示
                    ),
                    legendgroup='Left', # ドロップダウン制御用
                    visible=True # 初期表示はTrue
                ))
                added_left = True
        visible_left.append(added_left) # この球種の左打者トレースが追加されたか記録

        # 右打者データ
        if not df_right.empty:
            df_pt_right = df_right[df_right['TaggedPitchType'] == pt]
            if not df_pt_right.empty:
                custom_data_right = df_pt_right[['Pitcher', 'Batter', 'PlayResult', 'c_URL']].fillna('N/A').values
                fig.add_trace(go.Scattergl(
                    x=df_pt_right['PlateLocSide'],
                    y=df_pt_right['PlateLocHeight'],
                    mode='markers+text' if txt else 'markers',
                    marker=dict(color=color, size=ss, opacity=alpha, symbol=mark_right),
                    name=f"{pt} (Right)",
                    text=df_pt_right['PitchofPA'].apply(lambda x: int(round(x, 0)) if pd.notna(x) else '') if txt else None,
                    textposition='top center',
                    customdata=custom_data_right,
                    hovertemplate=(
                        "<b>Pitcher:</b> %{customdata[0]}<br>"
                        "<b>Batter:</b> %{customdata[1]}<br>"
                        "<b>Result:</b> %{customdata[2]}<br>"
                        "<b>URL:</b> %{customdata[3]}<br>"
                        "X: %{x:.2f}, Y: %{y:.2f}<extra></extra>"
                    ),
                    legendgroup='Right', # ドロップダウン制御用
                    visible=True # 初期表示はTrue
                ))
                added_right = True
        visible_right.append(added_right) # この球種の右打者トレースが追加されたか記録
    x_axis_range = [-1.0, 1.0] # m
    y_axis_range = [0.0, 2.0]  # m
    # --- レイアウト設定 ---
    fig.update_layout(
        title={'text': title, 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, # タイトル位置調整
        xaxis=dict(range=x_axis_range, showgrid=False, zeroline=False, title='PlateLocSide (m)'), # ラベル変更
        yaxis=dict(range=y_axis_range, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1, title='PlateLocHeight (m)'), # ラベル変更# 軸ラベル、アスペクト比固定
        width=650,
        height=650, # アスペクト比1:1に近づける
        showlegend=legend,
        # ★ 背景色を白に設定
        paper_bgcolor='white', # プロットエリア外の背景
        plot_bgcolor='white',  # プロットエリア内の背景
        xaxis_showticklabels=False, # X軸の目盛りラベル非表示
        yaxis_showticklabels=False, # Y軸の目盛りラベル非表示
    )

    # --- ストライクゾーンの描画 ---
    sz_top_ft = 1.05 # ストライクゾーン上端 (ft換算, 要調整)
    sz_bot_ft = 0.45 # ストライクゾーン下端 (ft換算, 要調整)
    sz_side_ft = 0.708 / 2 # ホームベース幅(17inch=1.416ft) / 2 + ボール半径？ (約0.354ft) - 要調整
                      # 一般的なSZの左右幅として使われる 0.708ft (8.5inch) を使うことが多い
                      # 参考: MLB Gamedayは約±0.83ft (ボール幅考慮)
    sz_left = -sz_side_ft
    sz_right = sz_side_ft

    # ゾーンをメートルからフィート換算する場合の参考値 (1m = 3.28084 ft)
    # sz_top_m = 1.05 * 3.28084 # 約3.44 ft
    # sz_bot_m = 0.45 * 3.28084 # 約1.48 ft
    # sz_side_m = 0.2159 * 3.28084 # 約0.708 ft (ホームベース幅/2)

    # 調整後の値 (一般的なフィート基準のSZ)
    # 一般的なSZ高さ: 膝～胸の中間あたり。打者によって変動するが固定値で描画する。
    # MLB GamedayのSZ (feet): x=[-0.83, 0.83], y=[1.5, 3.5] 付近が多い
    # ここでは元のコードの値を尊重しつつ、フィート単位で調整
    #sz_top_ft = 3.5 # 例: 胸の高さ付近
    #sz_bot_ft = 1.5 # 例: 膝の高さ付近
    #sz_side_ft = 0.83 # 例: ホームベース幅 + ボール左右分

    strike_zone_shapes = [
        # 実線ゾーン (MLB Gameday風)
        dict(type="rect", x0=-sz_side_ft, y0=sz_bot_ft, x1=sz_side_ft, y1=sz_top_ft, line=dict(color="black", width=2)),
        # 9分割線 (破線)
        dict(type="line", x0=sz_left + (sz_right-sz_left)/3, y0=sz_bot_ft, x1=sz_left + (sz_right-sz_left)/3, y1=sz_top_ft, line=dict(color="grey", width=1, dash="dash")),
        dict(type="line", x0=sz_left + 2*(sz_right-sz_left)/3, y0=sz_bot_ft, x1=sz_left + 2*(sz_right-sz_left)/3, y1=sz_top_ft, line=dict(color="grey", width=1, dash="dash")),
        dict(type="line", x0=sz_left, y0=sz_bot_ft + (sz_top_ft-sz_bot_ft)/3, x1=sz_right, y1=sz_bot_ft + (sz_top_ft-sz_bot_ft)/3, line=dict(color="grey", width=1, dash="dash")),
        dict(type="line", x0=sz_left, y0=sz_bot_ft + 2*(sz_top_ft-sz_bot_ft)/3, x1=sz_right, y1=sz_bot_ft + 2*(sz_top_ft-sz_bot_ft)/3, line=dict(color="grey", width=1, dash="dash")),
    ]

    # ★ Y軸のレンジをストライクゾーンに合わせて調整
    y_range_margin = 0.5 # 上下のマージン
    fig.update_yaxes(range=[sz_bot_ft - y_range_margin, sz_top_ft + y_range_margin])
    # ★ X軸のレンジもストライクゾーンに合わせて調整
    x_range_margin = 0.5
    fig.update_xaxes(range=[sz_left - x_range_margin, sz_right + x_range_margin])


    for shape in strike_zone_shapes:
        fig.add_shape(**shape, layer='below') # データの背面に描画


    # --- ドロップダウンメニュー (打者サイド切り替え) ---
    # fig.data には、まず左打者の全球種、次に右打者の全球種が順番に追加されていると仮定
    # visibleリストを使って、実際にトレースが存在するか確認
    num_pitch_types = len(pitch_types)
    vis_pattern_both = []
    vis_pattern_left = []
    vis_pattern_right = []

    for i in range(num_pitch_types):
        # 左打者のトレース
        if visible_left[i]:
            vis_pattern_both.append(True)
            vis_pattern_left.append(True)
            vis_pattern_right.append(False)
        # 右打者のトレース
        if visible_right[i]:
            vis_pattern_both.append(True)
            vis_pattern_left.append(False)
            vis_pattern_right.append(True)

    # データがない場合はドロップダウンを表示しない
    if not df_left.empty and not df_right.empty:
        updatemenus = [
            dict(
                type="dropdown",
                direction="down",
                x=0.05, # 位置調整
                y=1.15, # 位置調整
                xanchor='left',
                yanchor='top',
                showactive=True,
                buttons=list([
                    dict(label="左右打者", method="update", args=[{"visible": vis_pattern_both}, {"title": title + " (左右打者)"}]),
                    dict(label="左打者", method="update", args=[{"visible": vis_pattern_left}, {"title": title + " (左打者)"}]),
                    dict(label="右打者", method="update", args=[{"visible": vis_pattern_right}, {"title": title + " (右打者)"}])
                ]),
            )
        ]
        fig.update_layout(updatemenus=updatemenus)
    elif not df_left.empty:
         # 左打者のみデータがある場合
         fig.update_layout(title=title + " (左打者)")
    elif not df_right.empty:
         # 右打者のみデータがある場合
         fig.update_layout(title=title + " (右打者)")
    else:
         # 両方データがない場合（基本的には関数の最初で弾かれるはず）
         fig.update_layout(title=title + " (データなし)")

    if rtnfig:
        return fig
    else:
        # HTML出力する場合（現状のコードでは未使用だが残しておく）
        html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
        # JavaScriptは pitch_plot_page 側で `st.markdown` を使って注入する方式に変更
        return html_str


# CSV出力モード用のインタラクティブプロット関数
def create_interactive_plot_for_csv(df, df_left, df_right, title):
    """
    CSV出力モード用のインタラクティブプロットを作成する
    クリックされた点は、選択済みとして色が変わり、セッションに保存される
    Hover情報は指定されたカラムを表示する
    """
    required_plot_cols = ['TaggedPitchType','PlateLocSide','PlateLocHeight', 'BatterSide', 'Pitcher', 'Batter', 'PlayResult', 'c_URL', 'PitchofPA']
    if df.empty or not all(col in df.columns for col in required_plot_cols):
        st.warning("プロットに必要なデータが不足しているため、表示できません。")
        fig = go.Figure()
        fig.update_layout(title=title + " (データ不足)", xaxis={'visible': False}, yaxis={'visible': False})
        return fig

    # NaNを除外・置換
    df1 = df.dropna(subset=['PlateLocSide','PlateLocHeight']).copy()
    df1['TaggedPitchType'] = df1['TaggedPitchType'].fillna('Undefined')
    df1['PlayResult'] = df1['PlayResult'].fillna('Undefined')
    df1['Pitcher'] = df1['Pitcher'].fillna('Unknown')
    df1['Batter'] = df1['Batter'].fillna('Unknown')
    df1['c_URL'] = df1['c_URL'].fillna('') # c_URLは空文字に

    # 左右打者データの準備
    df_left = df1[df1['BatterSide'] == 'Left']
    df_right = df1[df1['BatterSide'] == 'Right']

    pitch_types = ['Fastball', 'Slider', 'Cutter', 'Curveball', 'ChangeUp', 'Splitter', 'Sinker', 'Knuckleball', 'TwoSeamFastBall', 'Other', 'Undefined']
    colors = ['red', 'limegreen', 'darkviolet', 'orange', 'deepskyblue', 'magenta', 'blue', 'silver', 'dodgerblue', 'black', 'grey']
    marks = [['circle', 'triangle-left', 'triangle-left', 'triangle-up', 'triangle-down', 'triangle-down', 'triangle-right', 'star', 'square', 'cross', 'x'],
             ['circle', 'triangle-right', 'diamond', 'triangle-up', 'diamond-wide', 'triangle-down', 'hexagon', 'star', 'hexagon2', 'cross', 'x']]

    fig = go.Figure()

    # すでに選択済みの投球インデックスリストを取得
    selected_indices = st.session_state.selected_pitches

    # --- トレースの追加 (選択/未選択で分ける) ---
    for pt, color, mark_left, mark_right in zip(pitch_types, colors, marks[0], marks[1]):
        for side, df_side, mark in [('Left', df_left, mark_left), ('Right', df_right, mark_right)]:
            if not df_side.empty:
                df_pt = df_side[df_side['TaggedPitchType'] == pt].copy() # df_pt をコピー

                if not df_pt.empty:
                    # hover表示に必要なカラムを追加 (fillnaもここで)
                    hover_cols = ['Pitcher', 'Batter', 'PlayResult', 'c_URL']
                    for hc in hover_cols:
                        if hc not in df_pt.columns: df_pt[hc] = 'N/A' # カラムなければ作成
                        else: df_pt[hc] = df_pt[hc].fillna('N/A') # 既存カラムのNaN処理

                    # 未選択の投球 (薄い色)
                    unselected = df_pt[~df_pt.index.isin(selected_indices)]
                    if not unselected.empty:
                        # hover_data用にリストのリストを作成
                        hover_data_unselected = unselected[hover_cols].values.tolist()
                        fig.add_trace(go.Scattergl(
                            x=unselected['PlateLocSide'],
                            y=unselected['PlateLocHeight'],
                            mode='markers',
                            marker=dict(color=color, size=10, opacity=0.3, symbol=mark),
                            name=f"{pt} ({side}) - 未選択",
                            customdata=unselected.index.tolist(), # クリックイベント用にインデックスを渡す
                            # hover_data=hover_data_unselected, # hover表示用データ
                            hovertemplate=( # hoverテンプレート (hover_dataを参照)
                                "<b>Pitcher:</b> %{customdata[0]}<br>" # hover_dataの代わりにdfの列を参照させる方法を試す
                                "<b>Batter:</b> %{customdata[1]}<br>"
                                "<b>Result:</b> %{customdata[2]}<br>"
                                "<b>URL:</b> %{customdata[3]}<br>"
                                "<b>Index:</b> %{customdata}<br>" # インデックス表示
                                "X: %{x:.2f}, Y: %{y:.2f}<extra></extra>"
                            ),
                            hoverinfo='skip', # デフォルトのhoverinfoは使わず、hovertemplateのみにする場合
                            meta=unselected[hover_cols].to_dict('records'), # 代替案: metaに辞書リスト格納
                        ))

                        # ↑ hovertemplateで %{hoverdata[i]} を使う代わりに、
                        # customdata に [index, Pitcher, Batter, Result, URL] のように複数情報入れる方法もあるが、
                        # クリックイベントでの index 取得が少し複雑になる。
                        # Plotly は df の列名を直接参照できる場合もある: "Pitcher: %{customdata[0]}"
                        # customdata に ['Pitcher', 'Batter', ...] を設定する必要がある
                        # 今回はシンプルに index を customdata に、他は hover_data に格納する方針で進めたが、
                        # 上記の代替案も検討の余地あり。
                        # → `meta` を使って `hovertemplate` で参照する方式に変更してみる
                        fig.update_traces(
                             hovertemplate=(
                                "<b>Pitcher:</b> %{meta[Pitcher]}<br>"
                                "<b>Batter:</b> %{meta[Batter]}<br>"
                                "<b>Result:</b> %{meta[PlayResult]}<br>"
                                "<b>URL:</b> %{meta[c_URL]}<br>"
                                "<b>Index:</b> %{customdata}<br>"
                                "X: %{x:.2f}, Y: %{y:.2f}<extra></extra>"
                            ), selector=dict(name=f"{pt} ({side}) - 未選択") # 直前に追加したトレースに適用
                         )


                    # 選択済みの投球 (濃い色)
                    selected = df_pt[df_pt.index.isin(selected_indices)]
                    if not selected.empty:
                        hover_data_selected = selected[hover_cols].values.tolist()
                        fig.add_trace(go.Scattergl(
                            x=selected['PlateLocSide'],
                            y=selected['PlateLocHeight'],
                            mode='markers',
                            marker=dict(color=color, size=10, opacity=1.0, symbol=mark,
                                        line=dict(color='black', width=1)), # 選択済みを強調
                            name=f"{pt} ({side}) - 選択済み",
                            customdata=selected.index.tolist(),
                            # hover_data=hover_data_selected,
                            hovertemplate=(
                                "<b>Pitcher:</b> %{meta[Pitcher]}<br>"
                                "<b>Batter:</b> %{meta[Batter]}<br>"
                                "<b>Result:</b> %{meta[PlayResult]}<br>"
                                "<b>URL:</b> %{meta[c_URL]}<br>"
                                "<b>Index:</b> %{customdata}<br>" # インデックス表示
                                "X: %{x:.2f}, Y: %{y:.2f}<extra></extra>"
                            ),
                             hoverinfo='skip',
                             meta=selected[hover_cols].to_dict('records'), # metaに格納
                        ))
                        fig.update_traces(
                             hovertemplate=(
                                "<b>Pitcher:</b> %{meta[Pitcher]}<br>"
                                "<b>Batter:</b> %{meta[Batter]}<br>"
                                "<b>Result:</b> %{meta[PlayResult]}<br>"
                                "<b>URL:</b> %{meta[c_URL]}<br>"
                                "<b>Index:</b> %{customdata}<br>"
                                "X: %{x:.2f}, Y: %{y:.2f}<extra></extra>"
                            ), selector=dict(name=f"{pt} ({side}) - 選択済み")
                         )

    # --- ストライクゾーンの描画 (通常モードと同じ設定を使用) ---
    sz_top_ft = 3.5
    sz_bot_ft = 1.5
    sz_side_ft = 0.83
    sz_left = -sz_side_ft
    sz_right = sz_side_ft

    strike_zone_shapes = [
        dict(type="rect", x0=sz_left, y0=sz_bot_ft, x1=sz_right, y1=sz_top_ft, line=dict(color="black", width=2)),
        dict(type="line", x0=sz_left + (sz_right-sz_left)/3, y0=sz_bot_ft, x1=sz_left + (sz_right-sz_left)/3, y1=sz_top_ft, line=dict(color="grey", width=1, dash="dash")),
        dict(type="line", x0=sz_left + 2*(sz_right-sz_left)/3, y0=sz_bot_ft, x1=sz_left + 2*(sz_right-sz_left)/3, y1=sz_top_ft, line=dict(color="grey", width=1, dash="dash")),
        dict(type="line", x0=sz_left, y0=sz_bot_ft + (sz_top_ft-sz_bot_ft)/3, x1=sz_right, y1=sz_bot_ft + (sz_top_ft-sz_bot_ft)/3, line=dict(color="grey", width=1, dash="dash")),
        dict(type="line", x0=sz_left, y0=sz_bot_ft + 2*(sz_top_ft-sz_bot_ft)/3, x1=sz_right, y1=sz_bot_ft + 2*(sz_top_ft-sz_bot_ft)/3, line=dict(color="grey", width=1, dash="dash")),
    ]
    for shape in strike_zone_shapes:
        fig.add_shape(**shape, layer='below')
    x_axis_range = [-1.0, 1.0] # m
    y_axis_range = [0.0, 2.0] 
    # --- プロットレイアウト ---
    fig.update_layout(
        title={'text': title + " (クリックで選択/解除)", 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis=dict(range=x_axis_range, showgrid=False, zeroline=False, title='PlateLocSide (m)'), # ラベル変更
        yaxis=dict(range=y_axis_range, showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1, title='PlateLocHeight (m)'),
        width=650,
        height=650,
        showlegend=False, # 凡例は非表示（選択/未選択で分かれているため）
        # ★ 背景色を白に設定
        paper_bgcolor='white',
        plot_bgcolor='white',
        clickmode='event', # クリックイベントを有効化
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
    )

    return fig

# コースプロットページの関数

# === テスト3a用: 複雑プロット表示 (plotly_events なし) ===
def pitch_plot_page_test3a():
    st.title("⚾ 投球コースプロット")

    # --- サイドバーと状態初期化 (元のロジックを復元) ---
    st.sidebar.header("⚙️ 表示設定")

    if 'selected_pitches' not in st.session_state:
        st.session_state.selected_pitches = []
    if 'filter_state' not in st.session_state:
        st.session_state.filter_state = {}
    if 'trackman_data' not in st.session_state:
        st.session_state.trackman_data = None
    if 'plot_generated' not in st.session_state:
        st.session_state.plot_generated = False
    # clicked_url_to_open はイベント処理無効なので不要
    # if 'clicked_url_to_open' not in st.session_state:
    #     st.session_state.clicked_url_to_open = None

    csv_mode = st.sidebar.checkbox("CSV出力モードを有効にする", key='csv_mode_check', value=st.session_state.get('csv_mode_check', False))

    st.sidebar.subheader("絞り込み条件")

    # データロード
    if st.session_state.trackman_data is None:
        with st.spinner("Trackmanデータを読み込んでいます..."):
            st.session_state.trackman_data = load_trackman_data()

    df_trackman = st.session_state.trackman_data

    if df_trackman is None or df_trackman.empty:
        st.error("Trackmanデータの読み込みに失敗しました。")
        st.stop()

    # --- フィルタUI (元のロジックを復元) ---
    filter_state = st.session_state.filter_state
    # (オプション取得部分は省略 - 前のコードを流用してください)
    pitcher_options = sorted(df_trackman["Pitcher"].dropna().unique().astype(str).tolist()) if "Pitcher" in df_trackman.columns else []
    batter_options = sorted(df_trackman["Batter"].dropna().unique().astype(str).tolist()) if "Batter" in df_trackman.columns else []
    balls_options = sorted(df_trackman["Balls"].dropna().unique().astype(int).tolist()) if "Balls" in df_trackman.columns else []
    strikes_options = sorted(df_trackman["Strikes"].dropna().unique().astype(int).tolist()) if "Strikes" in df_trackman.columns else []
    outs_options = sorted(df_trackman["Outs"].dropna().unique().astype(int).tolist()) if "Outs" in df_trackman.columns else []
    pitch_call_options = sorted(df_trackman["PitchCall"].dropna().unique().tolist()) if "PitchCall" in df_trackman.columns else []
    play_result_options = sorted(df_trackman["PlayResult"].dropna().unique().tolist()) if "PlayResult" in df_trackman.columns else []

    pitcher = st.sidebar.multiselect("投手名", pitcher_options, default=filter_state.get('pitcher', []))
    batter = st.sidebar.multiselect("打者名", batter_options, default=filter_state.get('batter', []))
    balls = st.sidebar.multiselect("ボール数", balls_options, default=filter_state.get('balls', []))
    strikes = st.sidebar.multiselect("ストライク数", strikes_options, default=filter_state.get('strikes', []))
    outs = st.sidebar.multiselect("アウト数", outs_options, default=filter_state.get('outs', []))
    pitch_calls = st.sidebar.multiselect("投球コール (PitchCall)", pitch_call_options, default=filter_state.get('pitch_calls', []))
    play_results = st.sidebar.multiselect("プレー結果 (PlayResult)", play_result_options, default=filter_state.get('play_results', []))

    plot_button = st.sidebar.button("📊 プロット表示", key="plot_button")

    # --- フィルター変更検知 (元のロジックを復元) ---
    # ★注意: もし以前このブロックをコメントアウトしてループが止まったなら、
    # このブロック内に問題がある可能性が高い。今回はテストのため一旦戻す。
    filter_changed = False
    current_filter_state = {
        'pitcher': pitcher, 'batter': batter, 'balls': balls, 'strikes': strikes,
        'outs': outs, 'pitch_calls': pitch_calls, 'play_results': play_results
    }
    if current_filter_state != st.session_state.filter_state:
        filter_changed = True
        st.session_state.filter_state = current_filter_state
        st.session_state.selected_pitches = []
        st.session_state.plot_generated = False

    # --- データフィルタリング (元のロジックを復元) ---
    filtered_df = df_trackman.copy()
    if pitcher: filtered_df = filtered_df[filtered_df["Pitcher"].isin(pitcher)]
    if batter: filtered_df = filtered_df[filtered_df["Batter"].isin(batter)]
    if balls: filtered_df = filtered_df[filtered_df["Balls"].isin(balls)]
    if strikes: filtered_df = filtered_df[filtered_df["Strikes"].isin(strikes)]
    if outs: filtered_df = filtered_df[filtered_df["Outs"].isin(outs)]
    if pitch_calls: filtered_df = filtered_df[filtered_df["PitchCall"].isin(pitch_calls)]
    if play_results: filtered_df = filtered_df[filtered_df["PlayResult"].isin(play_results)]

    # st.session_state.filtered_data = filtered_df # 必要なら復元

    # --- メイン表示エリア ---
    plot_container = st.container()

    with plot_container:
        # --- URLオープン処理は実行しない ---
        # if st.session_state.clicked_url_to_open: ... (削除)

        # --- プロット表示条件 (元のロジックを復元) ---
        should_plot = plot_button or filter_changed or not st.session_state.plot_generated

        if not filtered_df.empty and should_plot:
            st.session_state.plot_generated = True
            st.write(f"表示データ数: {len(filtered_df)}件")

            # 左右打者分割 (元のロジック)
            if 'BatterSide' in filtered_df.columns:
                df_left = filtered_df[filtered_df['BatterSide'] == 'Left']
                df_right = filtered_df[filtered_df['BatterSide'] == 'Right']
            else:
                st.warning("'BatterSide' カラムが見つかりません。")
                df_left = pd.DataFrame(columns=filtered_df.columns)
                df_right = filtered_df.copy()

            # タイトル生成 (元のロジック)
            plot_title_parts = ["投球コース"]
            if pitcher: plot_title_parts.append(f"投手: {', '.join(pitcher)}")
            if batter: plot_title_parts.append(f"打者: {', '.join(batter)}")
            plot_title = " | ".join(plot_title_parts)

            # --- モード別プロット表示 (plotly_events なし) ---
            if csv_mode:
                st.subheader("📈 コースプロット (CSV選択モード)")
                try:
                    fig = create_interactive_plot_for_csv(filtered_df, df_left, df_right, plot_title)
                    st.plotly_chart(fig, use_container_width=True) # ★ 表示のみ
                    # --- plotly_events と関連処理はコメントアウト ---
                    # selected_points = plotly_events(fig, ...)
                    # if selected_points: ...
                    # st.write(f"選択済み...")
                    # col1, col2 = st.columns(2) ... (ボタン類)
                except Exception as e:
                    st.error(f"CSVモードのプロット表示中にエラー: {e}")

            else:
                # === 通常モード ===
                st.subheader("📈 コースプロット (通常モード)")
                try:
                    # rtnfig=True を想定
                    fig = BTZplot_plotly(filtered_df, df_left, df_right, title=plot_title, rtnfig=True)
                    st.plotly_chart(fig, use_container_width=True) # ★ 表示のみ
                    # --- plotly_events と関連処理はコメントアウト ---
                    # clicked_data = plotly_events(fig, ...)
                    # if clicked_data: ... (URL処理など)
                except Exception as e:
                    st.error(f"通常モードのプロット表示中にエラー: {e}")

        elif should_plot and filtered_df.empty:
            st.warning("指定されたフィルタ条件に一致するデータがありません。条件を変更して再試行してください。")
        elif not should_plot:
             st.info("← サイドバーで条件を選択し、「プロット表示」ボタンをクリックしてください。")