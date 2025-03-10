import streamlit as st
import pandas as pd
import requests
import base64
import io
import urllib.parse
import unicodedata
import time
from datetime import datetime
import concurrent.futures

# Googleスプレッドシートからデータを取得するためのライブラリ
from google.oauth2 import service_account
import gspread

# baseballmetricsモジュール（成績計算関数群）をインポート
from baseballmetrics import *  
import glob
#@st.cache_data(ttl=86640)
#def load_data_from_google_spreadsheet():
#    """
#    st.secretsに設定したGoogleのサービスアカウント情報と
#    スプレッドシートキーをもとに、対象のスプレッドシートの全ワークシートの内容を
#    DataFrameに読み込み、結合して返す関数。
#    """
#    # Secretsから認証情報とスプレッドシートキーを取得
#    credentials_info = st.secrets["gcp_service_account"]
#    spreadsheet_key = st.secrets["gspread"]["spreadsheet_key"]
#    scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
#    credentials = service_account.Credentials.from_service_account_info(credentials_info, scopes=scopes)
#    
#    # gspreadを使ってスプレッドシートに接続
#    gc = gspread.authorize(credentials)
#    try:
#        sh = gc.open_by_key(spreadsheet_key)
#    except Exception as e:
#        st.error("スプレッドシートの読み込みに失敗しました。Secretsの設定とスプレッドシートキーを確認してください。")
#        st.exception(e)
#        return None
#
#    dfs = []
#    # 各ワークシートのデータを取得してDataFrameに変換
#    for worksheet in sh.worksheets():
#        data = worksheet.get_all_values()
#        if len(data) < 2:
#            continue  # ヘッダーのみまたは空のシートはスキップ
#        df_sheet = pd.DataFrame(data[1:], columns=data[0])
#        dfs.append(df_sheet)
#    if dfs:
#        combined_df = pd.concat(dfs, ignore_index=True)
#        return combined_df
#    else:
#        return None

@st.cache_data(ttl=86640)
def load_data_from_csv():
    """
    ローカルのCSVファイルからデータを読み込む関数
    """
    try:
        # CSVファイルのパスを指定（複数ある場合はリストで）
        S23 = glob.glob(f'data/*.csv')
        df23s = pd.DataFrame()
        for fl in S23:
            df0 = pd.read_csv(fl, encoding = 'utf-8-sig')
            df23s = pd.concat([df23s, df0]).reset_index(drop = True)
        return df23s
    except Exception as e:
        st.error(f"CSVファイルの読み込みに失敗しました: {e}")
        return None
    

def compute_batter_stats(df):
    """
    結合したdfを"Batter"ごとにグループ化し、各グループに対して
    指定された関数（seki, dasu, countpr, BA, OBP, SA, OPS）を適用して集計結果を返す。
    
    戻り値は各指標の列名を含むDataFrameで、列順は以下の通り:
    ['Batter', '打席', '打数', '安打', '二塁打', '三塁打', '本塁打', '四球', '三振', '打率', '出塁率', '長打率', 'OPS']
    """
    df = df.copy()
    df= df.query('BatterTeam == "TOK"').reset_index(drop=True)
    results = []
    # "Batter"という列でグループ化
    for batter, group in df.groupby("Batter"):
        # 各関数を適用（関数内部でgroupのデータを使って計算する前提）
        single = countpr(group, 'Single')
        double = countpr(group, 'Double')
        triple = countpr(group, 'Triple')
        homerun = countpr(group, 'HomeRun')
        stats = {
            "打者": batter,
            "打席": seki(group),
            "打数": dasu(group),
            "安打": single + double + triple + homerun,
            "単打": countpr(group, 'Single'),
            "二塁打": countpr(group, 'Double'),
            "三塁打": countpr(group, 'Triple'),
            "本塁打": countpr(group, 'HomeRun'),
            "打点": int(group.RunsScored.sum()),
            "犠打": countpr(group.query('TaggedHitType == "GroundBall"'), 'Sacrifice'),
            "犠飛": countpr(group.query('TaggedHitType == "FlyBall"'), 'Sacrifice'),
            "四球": countpr(group, 'Walk'),
            "三振": countpr(group, 'Strikeout'),
            "打率": BA(group),
            "出塁率": OBP(group,mc=False),
            "長打率": SA(group),
            "OPS": OPS(group),
            "三振率": round(countpr(group, 'Strikeout') / seki(group)*100,1),
        }
        results.append(stats)
    
    # 結果をDataFrameに変換
    result_df = pd.DataFrame(results)
    # 表示順を整える
    result_df = result_df[["打者",  "打率", "打点","出塁率", "長打率", "OPS","打席", "打数", "安打","単打", "二塁打", "三塁打", "本塁打", "四球", "三振","犠打","犠飛","三振率"]].sort_values("OPS", ascending=False)
    return result_df

def calculate_stats_pitcher(sheet_0222):
    def calculate_inning_from_float(num):
        try:
            seisu = int(num)
        except:
            return '-'
        shosu = round(num - seisu,2)
        if shosu < 0.3:
            return str(seisu)
        elif 0.3 <= shosu <=0.6:
            shosu = '1/3'
        else:
            shosu = '2/3'
        return str(seisu) + ' ' + shosu

    result = []
    for df in sheet_0222.groupby('Pitcher'):
        result_moto = []
        df_1 = df[1]
        num_runner = countpr(df_1,'Single') + countpr(df_1,'Double') + countpr(df_1,'Triple') + countpr(df_1,'HomeRun') + countpr(df_1,'Walk') + countpr(df_1,'HitByPitch')
        df_pitches = df_1.dropna(subset=['PitchofPA'])
        ph = ['InPlay', 'HitByPitch'] # 変数
        kb = ['Strikeout', 'Walk'] # 変数
        df_kekka = df_1.query('PitchCall == @ph or KorBB == @kb')
        ini = (int(df_1.OutsOnPlay.sum()) + len(df_1.query('KorBB == "Strikeout"'))) / 3
        ini_str = calculate_inning_from_float(ini)
        result_moto.append(df[0])
        result_moto.append(ini_str)
        result_moto.append(seki(df_1))
        result_moto.append(int(df_1.RunsScored.sum()))
        result_moto.append(BA(df_1,mc=True)[2])
        result_moto.append(len(df_1.query('KorBB == "Strikeout"')))
        result_moto.append(len(df_1.query('KorBB == "Walk"')))
        result_moto.append(round(len(df_pitches.query('PitchCall != "BallCalled"'))/len(df_pitches),2))
        result_moto.append(round((len(df_1.query('KorBB == "Strikeout"')) - len(df_1.query('KorBB == "Walk"'))) / seki(df_1)*100,1))
        result_moto.append(BA(df_1,mc=False))
        result_moto.append(OPS(df_1))
        result_moto.append(round(len(df_1.query('KorBB == "Strikeout"')) / seki(df_1)*100,1))
        result_moto.append(round(len(df_1.query('KorBB == "Walk"')) / seki(df_1)*100,1))
        result_moto.append(round(len(df_1.query('TaggedHitType == "GroundBall" and PitchCall == "InPlay"')) / seki(df_1)*100,1))
        result_moto.append(round(len(df_1.query('TaggedHitType == "LineDrive" and PitchCall == "InPlay"')) / seki(df_1)*100,1))
        result_moto.append(round(len(df_1.query('TaggedHitType == "FlyBall" and PitchCall == "InPlay"')) / seki(df_1)*100,1))
        result_moto.append(round(num_runner/ini,2))

        result.append(result_moto)
    result_final = pd.DataFrame(result, columns = ['Pitcher', '回', '打者', '点','安','K', 'BB', 'ストライク率','K-BB%','被打率','被OPS','K%','BB%','GB%','LD%','FB%','WHIP'])
    return result_final


def main():
    st.title("オープン戦成績")
    st.write("現在の対象試合:2025春 日付があっていない場合は左のバーにあるデータ更新ボタンを押してください。")
    st.subheader("操作方法")
    st.write("左のサイドバーで、データ種別（打者成績 or 投手成績）やGameLevel（A戦 or B戦）を選択してください。")
    st.write("また、日付範囲をスライダーで選択することで、指定した日付範囲のデータを表示できます。")
    st.write("※初回読み込みには1分程度かかります。原因調査中です。")
    
    # ここでGoogleスプレッドシートからデータを取得し、st.cache_dataでキャッシュ
    df = load_data_from_csv()
    if df is None:
        st.error("データが取得できませんでした。")
        return
        # 更新ボタンをサイドバーに配置
    update_button = st.sidebar.button("データ更新")
    if update_button:
        st.info("更新ボタンが押されたので、キャッシュをクリアして最新データを取得します。")
        # st.cache_dataのキャッシュをクリアして再読み込み
        load_data_from_csv.clear()
        data = load_data_from_csv()
        st.session_state.raw_data = data
    elif "raw_data" not in st.session_state:
        data = load_data_from_csv()
        st.session_state.raw_data = data
    else:
        st.info("キャッシュ済みのデータを使用しています。")
    # 以降は、前回のコード例と同様にフィルタ処理等を実施
    type_choice = st.sidebar.radio("データ種別を選択", ["打者成績", "投手成績"])
    # GameLevelフィルタ
    st.sidebar.subheader("GameLevel")
    A_checked = st.sidebar.checkbox("A", value=True)
    B_checked = st.sidebar.checkbox("B", value=True)
    if A_checked and not B_checked:
        df = df[df["Level"] == "A"]
    elif B_checked and not A_checked:
        df = df[df["Level"] == "B"]
    elif not A_checked and not B_checked:
        st.warning("少なくとも一方のGameLevel（A戦またはB戦）を選択してください。")
        return
    if len(df) == 0:
        st.warning("該当するデータがありません。")
        return
            # Dateフィルタ（スライダー）
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    min_date = df["Date"].min().to_pydatetime()
    max_date = df["Date"].max().to_pydatetime()
    selected_date_range = st.slider("日付範囲を選択", min_value=min_date, max_value=max_date,
                                    value=(min_date, max_date), format="YYYY-MM-DD")
    df = df[(df["Date"] >= selected_date_range[0]) & (df["Date"] <= selected_date_range[1])]
    # Date列は集計に不要なので削除
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    if type_choice == "打者成績":
    # PitcherThrowsフィルタ
        st.sidebar.subheader("PitcherThrows")
        right_checked = st.sidebar.checkbox("Right", value=True)
        left_checked = st.sidebar.checkbox("Left", value=True)
        if right_checked and not left_checked:
            df = df[df["PitcherThrows"] == "Right"]
        elif left_checked and not right_checked:
            df = df[df["PitcherThrows"] == "Left"]
        elif not right_checked and not left_checked:
            st.warning("少なくとも一方の投球方向（RightまたはLeft）を選択してください。")
            return
        if len(df) == 0:
            st.warning("該当するデータがありません。")
            return
        try:
            stats_df = compute_batter_stats(df)
            st.subheader("打者成績")
            st.dataframe(stats_df)
        except Exception as e:
            st.error("成績集計に失敗しました")
            st.exception(e)

    elif type_choice == "投手成績":
        st.sidebar.subheader("BatterSide")
        right_checked = st.sidebar.checkbox("Right", value=True)
        left_checked = st.sidebar.checkbox("Left", value=True)
        if right_checked and not left_checked:
            df = df[df["BatterSide"] == "Right"]
        elif left_checked and not right_checked:
            df = df[df["BatterSide"] == "Left"]
        elif not right_checked and not left_checked:
            st.warning("少なくとも一方の打席（RightまたはLeft）を選択してください。")
            return
        if len(df) == 0:
            st.warning("該当するデータがありません。")
            return
        try:
            df['OutsOnPlay'] = df['OutsOnPlay'].astype(int)
            df['RunsScored'] = df['RunsScored'].astype(int)
            stats_df = calculate_stats_pitcher(df.query('PitcherTeam == "TOK"').reset_index(drop=True)).sort_values('K-BB%', ascending=False)
                # すべてのint型の列をfloat型にキャストして、オーバーフローエラーを回避
            st.subheader("投手成績")
            st.dataframe(stats_df)
        except Exception as e:
            st.error("成績集計に失敗しました")
            st.exception(e)

if __name__ == "__main__":
    main()
