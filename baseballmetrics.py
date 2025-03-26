import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc, matplotlib.patches as pat
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import copy, math, io, os, random, requests , joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xwOBA_model.pkl")


def BA(df, mc=False):  # Batting Average
    hit = ["Single", "Double", "Triple", "HomeRun"]  # hitっていう変数作ります。
    df1 = df.reset_index(drop=True)
    ph = ["InPlay", "HitByPitch"]  # 変数
    kb = ["Strikeout", "Walk"]  # 変数
    df2 = df1.query("PitchCall == @ph or KorBB == @kb")
    ds = len(df2)  # 打席数
    sc = len(df2.query('PlayResult == "Sacrifice"')) + len(
        df2.query('TaggedHitType == "Bunt" and PlayResult == "FieldersChoice"')
    )
    # 犠牲打数
    ht = len(df2.query("PlayResult == @hit"))  # 安打数
    bb = len(df2.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))  # 四死球数
    if ds - sc - bb != 0:
        ba = round(ht / (ds - sc - bb), 3)  # 打率
    else:
        ba = np.nan  # 分母が0の時の打率
    if mc:
        return ba, ds - sc - bb, ht
    else:
        return ba


def HardHit(df, border=140, mc=False, name=None):
    if name != None:
        df = df.query("Batter == @name").reset_index(drop=True)
    df1 = df.reset_index(drop=True)
    ph = ["InPlay", "HitByPitch"]  # 変数
    kb = ["Strikeout", "Walk"]  # 変数
    df2 = df1.query("PitchCall == @ph or KorBB == @kb")
    ds = len(df2)  # 打席数
    sc = len(df2.query('PlayResult == "Sacrifice"')) + len(
        df2.query('TaggedHitType == "Bunt" and PlayResult == "FieldersChoice"')
    )
    # 犠牲打数
    ht = len(df2.query("ExitSpeed >= @border"))  # 安打数
    bb = len(df2.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))  # 四死球数
    if ds - sc - bb != 0:
        ba = round(ht / (ds - sc - bb), 3)  # 打率
    else:
        ba = np.nan  # 分母が0の時の打率
    if mc:
        return ba, ds - sc - bb, ht
    else:
        return ba


def OBP(df, mc=False):  # On-base Percentage
    hit = ["Single", "Double", "Triple", "HomeRun"]  # hitっていう変数作ります。

    df1 = df.reset_index(drop=True)
    ph = ["InPlay", "HitByPitch"]  # 同じく変数です。
    kb = ["Strikeout", "Walk"]  # 同じく変数です。
    bunt = ["Bunt", "GroundBall"]  # バントっていう変数
    df2 = df1.query("PitchCall == @ph or KorBB == @kb")

    ds = len(df2)  # 打席数
    ht = len(df2.query("PlayResult == @hit"))  # 安打数
    bb = len(df2.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))  # 四死球数
    bu = len(
        df2.query('PlayResult == "Sacrifice"').query("TaggedHitType == @bunt")
    )  # バント数
    if ds - bu != 0:
        obp = round((ht + bb) / (ds - bu), 3)  # 出塁率
    else:
        obp = np.nan  # 分母が0の時の出塁率
    if mc:
        return obp, ds - bu, ht + bb
    else:
        return obp


def SA(df, mc=False):  # Slugging Average
    df1 = df.reset_index(drop=True)
    ph = ["InPlay", "HitByPitch"]  # 同じく変数です。
    kb = ["Strikeout", "Walk"]  # 同じく変数です。
    df2 = df1.query("PitchCall == @ph or KorBB == @kb")
    ds = len(df2)  # 打席数
    sg = len(df2.query('PlayResult == "Single"'))  # 単打数
    two = len(df2.query('PlayResult == "Double"'))  # 2塁打数
    thr = len(df2.query('PlayResult == "Triple"'))  # 3塁打数
    hr = len(df2.query('PlayResult == "HomeRun"'))  # 本塁打数
    sc = len(df2.query('PlayResult == "Sacrifice"'))  # 犠牲打数
    bb = len(df2.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))  # 四死球数
    if ds - sc - bb != 0:
        sa = round((sg + 2 * two + 3 * thr + 4 * hr) / (ds - sc - bb), 3)  # 長打率
    else:
        sa = np.nan  # 分母が0の時の長打率
    if mc:
        return sa, ds - sc - bb, sg + 2 * two + 3 * thr + 4 * hr
    else:
        return sa


def HardHit(df, border=140, mc=False):  # Batting Averagehitっていう変数作ります。
    df1 = df.reset_index(drop=True)
    ph = ["InPlay", "HitByPitch"]  # 変数
    kb = ["Strikeout", "Walk"]  # 変数
    df2 = df1.query("PitchCall == @ph or KorBB == @kb")
    ds = len(df2)  # 打席数
    sc = len(df2.query('PlayResult == "Sacrifice"')) + len(
        df2.query('TaggedHitType == "Bunt" and PlayResult == "FieldersChoice"')
    )
    # 犠牲打数
    ht = len(df2.query("ExitSpeed >= @border"))  # 安打数
    bb = len(df2.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))  # 四死球数
    if ds - sc - bb != 0:
        ba = round(ht / (ds - sc - bb), 3)  # 打率
    else:
        ba = np.nan  # 分母が0の時の打率
    if mc:
        return ba, ds - sc - bb, ht
    else:
        return ba * 100


def OPS(df):  # On Base plus Slugging
    df1 = df.reset_index(drop=True)
    if OBP(df1, mc=True)[1] == 0:
        ops = SA(df1, mc=False)
    elif SA(df1, mc=True)[1] == 0:
        ops = OBP(df1, mc=False)
    else:
        ops = SA(df1, mc=False) + OBP(df1, mc=False)
    return round(ops, 3)


def IsoP(df):  # Isolated Power
    return round(SA(df) - BA(df), 3)


def IsoD(df):  # Isolated Discipline
    if OBP(df, mc=True)[1] == 0:
        isod = -BA(df)
    elif BA(df, mc=True)[1] == 0:
        isod = OBP(df)
    else:
        isod = round(OBP(df) - BA(df), 3)
    return isod


def BABIP(df, mc=False):  # Batting Average on Balls In Play
    hit = ["Single", "Double", "Triple", "HomeRun"]  # hitっていう変数作ります。

    df1 = df.reset_index(drop=True)
    ph = ["InPlay", "HitByPitch"]  # 同じく変数です。
    kb = ["Strikeout", "Walk"]  # 同じく変数です。
    bunt = ["Bunt", "GroundBall"]  # バントっていう変数
    df2 = df1.query("PitchCall == @ph or KorBB == @kb")

    ds = len(df2)  # 打席数
    ht = len(df2.query("PlayResult == @hit"))  # 安打数
    hr = len(df2.query('PlayResult == "HomeRun"'))  # ホームラン数
    bb = len(df2.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))  # 四死球数
    bu = len(
        df2.query('PlayResult == "Sacrifice" and TaggedHitType == @bunt')
    )  # バント数
    kk = len(df2.query('KorBB == "Strikeout"'))  # 三振数

    if ds - bb - kk - hr - bu != 0:
        bapip = round((ht - hr) / (ds - bb - kk - hr - bu), 3)
    else:
        bapip = np.nan
    if mc:
        return bapip, ds - bb - kk - hr - bu, ht - hr
    else:
        return bapip


def wOBA(df, coef=[0.7, 0.9, 1.3, 1.6, 2.0]):  # Weighted On Base Average
    df1 = df.reset_index(drop=True)
    ph = ["InPlay", "HitByPitch"]  # 同じく変数です。
    kb = ["Strikeout", "Walk"]  # 同じく変数です。
    bunt = ["Bunt", "GroundBall"]  # バントっていう変数
    df2 = df1.query("PitchCall == @ph or KorBB == @kb")
    ds = len(df2)  # 打席数
    bb = len(df2.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))  # 四死球数
    bu = len(
        df2.query('PlayResult == "Sacrifice"').query("TaggedHitType == @bunt")
    )  # バント数
    sg = len(df2.query('PlayResult == "Single"'))  # 単打数
    two = len(df2.query('PlayResult == "Double"'))  # 2塁打数
    thr = len(df2.query('PlayResult == "Triple"'))  # 3塁打数
    hr = len(df2.query('PlayResult == "HomeRun"'))  # 本塁打数

    if ds - bu != 0:
        woba = round(
            (coef[0] * bb + coef[1] * sg + coef[2] * two + coef[3] * thr + coef[4] * hr)
            / (ds - bu),
            3,
        )
    else:
        woba = np.nan
    return woba


def b6_wOBA(df):
    return wOBA(df, coef=[0.744, 0.938, 1.444, 2.074, 2.244])


def T_wOBA(df):  # 東大の得点価値に基づいたwOBA
    return wOBA(df, coef=[0.361, 0.457, 0.911, 1.487, 1.517])


def SWING(df, mc=False, azo=0, xrange=[-0.3, 0.3], zrange=[0.45, 1.05]):
    # azoは, AllかZoneかOut-Swing％のどれを出すかってことです。
    # 0だと普通の, 1だとZone-Swing%, 2だとOut-Swing%, 3以上だと全て出します。
    df1 = df.reset_index(drop=True)
    casw, masw, czsw, mzsw, cosw, mosw = np.zeros(6)

    nx, xx = xrange[0], xrange[1]
    nz, xz = zrange[0], zrange[1]

    swing = [
        "InPlay",
        "FoulBall",
        "StrikeSwinging",
        "FoulBallNotFieldable",
        "FoulBallFieldable",
    ]
    masw, casw = len(df1), len(df1.query("PitchCall == @swing"))
    if "PlateLocSide" in df1.columns:
        dfz = df1.query("@nx <= PlateLocSide <= @xx and @nz <= PlateLocHeight <= @xz")
        mzsw, czsw = len(dfz), len(dfz.query("PitchCall == @swing"))
        mosw, cosw = masw - mzsw, casw - czsw

    msw = [masw, mzsw, mosw]
    csw = [casw, czsw, cosw]

    if masw == 0:
        sw = [np.nan, np.nan, np.nan]
    elif mzsw == 0:
        sw = [round(casw / masw * 100, 1), np.nan, round(cosw / mosw * 100, 1)]
    elif mosw == 0:
        sw = [round(casw / masw * 100, 1), round(czsw / mzsw * 100, 1), np.nan]
    else:
        sw = [
            round(casw / masw * 100, 1),
            round(czsw / mzsw * 100, 1),
            round(cosw / mosw * 100, 1),
        ]

    if azo not in [0, 1, 2]:
        return sw
    else:
        if mc:
            return sw[azo], msw[azo], csw[azo]
        else:
            return sw[azo]


def WHIFF(df, mc=False, azo=0, xrange=[-0.3, 0.3], zrange=[0.45, 1.05]):
    # strikezoneのxの範囲とzの範囲を指定
    # azoは all, zone, out の切り替え設定 0か1か2
    df1 = df.reset_index(drop=True)
    cawf, mawf, czwf, mzwf, cowf, mowf = np.zeros(6)
    nx, xx = xrange[0], xrange[1]
    nz, xz = zrange[0], zrange[1]

    swing = [
        "InPlay",
        "FoulBall",
        "StrikeSwinging",
        "FoulBallNotFieldable",
        "FoulBallFieldable",
    ]
    mawf, cawf = len(df1.query("PitchCall == @swing")), len(
        df1.query('PitchCall == "StrikeSwinging"')
    )

    if "PlateLocSide" in df1.columns:
        dfz = df1.query(
            "@nx <= PlateLocSide <= @xx and @nz <= PlateLocHeight <= @xz"
        )  # ゾーンのz
        mzwf, czwf = len(dfz.query("PitchCall == @swing")), len(
            dfz.query('PitchCall == "StrikeSwinging"')
        )
        mowf, cowf = mawf - mzwf, cawf - czwf

    mwf = [mawf, mzwf, mowf]
    cwf = [cawf, czwf, cowf]

    if mawf == 0:
        wf = [np.nan, np.nan, np.nan]
    elif mzwf == 0:
        wf = [round(cawf / mawf * 100, 1), np.nan, round(cowf / mowf * 100, 1)]
    elif mowf == 0:
        wf = [round(cawf / mawf * 100, 1), round(czwf / mzwf * 100, 1), np.nan]
    else:
        wf = [
            round(cawf / mawf * 100, 1),
            round(czwf / mzwf * 100, 1),
            round(cowf / mowf * 100, 1),
        ]

    if azo not in [0, 1, 2]:
        return wf
    else:
        if mc:
            return wf[azo], mwf[azo], cwf[azo]
        else:
            return wf[azo]


def gpf(df, mc=False):  # goro per fly
    df = df.query('TaggedHitType != "Bunt"')
    df1 = df.query('PitchCall == "InPlay"').query("0 < Angle <= 15")
    df2 = df.query('PitchCall == "InPlay"').query("Angle <= 0")
    df3 = df.query('PitchCall == "InPlay"').query("Angle > 15")
    r = np.array(df1.Distance)
    th = np.array(df1.Direction)
    g, f = len(df2), len(df3)
    for i in range(len(df1)):
        x = r[i] * np.cos(th[i])
        y = r[i] * np.sin(th[i])
        if (y <= x) and (y >= -x) and (y <= -x + 38.795) and (y >= x - 38.795):
            g += 1
        else:
            f += 1
    if f != 0:
        if mc:
            return round(g / f, 2), f, g
        else:
            return round(g / f, 2)
    else:
        if mc:
            return f"{g}/{f}", f, g
        else:
            return f"{g}/{f}"


def changeba(Ba):
    if type(Ba) == str:
        return "-"
    if np.isnan(Ba):
        return "-"
    ba = round(round(Ba, 3) * 1000)
    if ba >= 1000:
        return f"{ba//1000}{changeba((ba%1000)/1000)}"
    elif Ba < 0.01:
        return f".00{str(ba)}"
    elif Ba < 0.1:
        return f".0{str(ba)}"
    else:
        return f".{str(ba)}"


def dasu(df):
    ph = ["InPlay", "HitByPitch"]  # 変数
    kb = ["Strikeout", "Walk"]  # 変数
    sc = len(df.query('PlayResult == "Sacrifice"'))  # 犠牲打数
    bb = len(df.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))  # 四死球数
    return len(df.query("PitchCall == @ph or KorBB == @kb")) - sc - bb


def seki(df):
    ph = ["InPlay", "HitByPitch"]  # 変数
    kb = ["Strikeout", "Walk"]  # 変数
    return len(df.query("PitchCall == @ph or KorBB == @kb"))


def countpr(df, pr="Single"):
    ph = ["InPlay", "HitByPitch"]  # 変数
    kb = ["Strikeout", "Walk"]  # 変数
    if pr in kb:
        return len(df.query("KorBB == @pr"))
    elif pr == "HitByPitch":
        return len(df.query("PitchCall == @pr"))
    else:
        return len(
            df.query("PitchCall == @ph or KorBB == @kb").query("PlayResult == @pr")
        )


def stl(df):
    if "runevent" in df.columns:
        return len(
            df.query(
                'runevent.str.contains("Steal", na=False)', engine="python"
            ).runevent
        ) - len(
            df.query(
                'runevent.str.contains("StealOut", na=False)', engine="python"
            ).runevent
        ), len(
            df.query(
                'runevent.str.contains("StealOut", na=False)', engine="python"
            ).runevent
        )
    elif "Steal" in df.columns:
        return len(df.query('Steal == "Steal"')), len(df.query('Steal == "StealOut"'))
    else:
        return 0, 0


# ws 0で全体、1で見逃し、2で空振り
def Kp(df, mc=False, ws=0):
    ds = dasu(df)
    if ws == 0:
        kk = len(df.query('KorBB == "Strikeout"'))
    elif ws == 1:
        kk = len(df.query('KorBB == "Strikeout" and PitchCall == "StrikeCalled"'))
    elif ws == 2:
        kk = len(df.query('KorBB == "Strikeout" and PitchCall == "StrikeSwinging"'))
    if ds == 0:
        kp = np.nan
    else:
        kp = round(kk * 100 / ds, 1)
    if mc:
        return kp, ds, kk
    else:
        return kp


def BBp(df, mc=False):
    pa = seki(df)
    bb = len(df.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))
    if pa == 0:
        bbp = np.nan
    else:
        bbp = round(bb * 100 / pa, 1)
    if mc:
        return bbp, pa, bb
    else:
        return bbp


def strikeratio2(df, mc=False):
    df = df.query('runevent != "PickOff"')
    strike = [
        "InPlay",
        "FoulBall",
        "StrikeSwinging",
        "StrikeCalled",
        "FoulBallFieldable",
        "FoulBallNotFieldable",
    ]
    df1 = df.query("PitchCall == @strike")
    if len(df) == 0:
        return "-"
    else:
        sr = round(len(df1) / len(df) * 100, 1)
        if mc:
            return f"{sr} {len(df1)}/{len(df)}"
        else:
            return sr


def retband(df, item="RelSpeed", kakko=False):
    if len(df) == 0:
        return "---"
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)
    if item not in df.describe().columns:
        return "---"
    result = df.describe()[item][["min", "mean", "max"]]
    if len(df) == 1:
        if kakko:
            return "[" + str(round(result.min())) + "]"
        else:
            return round(result.min())
    n = round(result.min())
    x = round(result.max())
    mn = round(result.mean())
    if kakko:
        return "(" + str(n) + "-" + str(x) + ")" + "[" + str(mn) + "]"
    else:
        return str(n) + "-" + str(x) + "\n" + "[" + str(mn) + "]"


def fip(df1):
    hr = len(df1.query('PlayResult == "HomeRun"'))  # HR数
    kk = len(df1.query('KorBB == "Strikeout"'))  # 奪三振数
    bb = len(df1.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))  # 四死球数
    inn = (
        (
            len(df1.query("OutsOnPlay == 1"))
            + len(df1.query("OutsOnPlay == 2")) * 2
            + len(df1.query("OutsOnPlay == 3"))
        )
        / 3
        * 3
    )  # 投球回
    if inn != 0:
        fip = round((13 * hr + 3 * bb - 2 * kk) / inn + 3, 2)
    else:
        fip = 99.99
    return fip


def RA(df):  # Runs Average
    runs_allowed = df["RunsScored"].sum()
    outs = df["OutsOnPlay"].sum()
    if outs == 0:
        return None
    else:
        return runs_allowed * 27 / outs


def tRA(df1):
    hr = len(df1.query('PlayResult == "HomeRun"'))  # HR数
    kk = len(df1.query('KorBB == "Strikeout"'))  # 奪三振数
    bb = len(df1.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))  # 四死球数
    gb = len(df1.query('AutoHitType == "GroundBall"'))  # ゴロ数
    lb = len(df1.query('AutoHitType == "LineDrive"'))  # ライナー数
    pb = len(df1.query('AutoHitType == "Popup"'))  # 打ち上げ数
    fb = len(df1.query('AutoHitType == "FlyBall"'))  # フライ数
    RuEx = (
        0.3 * bb
        + -0.11 * kk
        + 1.41 * hr
        + 0.03 * gb
        + -0.12 * pb
        + 0.12 * fb
        + 0.33 * lb
    )
    OuEx = kk + 0.75 * gb + -0.99 * pb + 0.71 * fb + 0.26 * lb
    tRA = round(27 * RuEx / OuEx, 2)
    return tRA


# 以下の計算ではb6をTrueにすると、20S-24Sの六大の得点期待値に基づいた打撃貢献を計算します(FalseだとNPB由来？)


def wRAA(df, ave_woba=0.33, b6=False, name=None):  # weighted Runs Above Average
    if name != None:
        df1 = df.query("Batter == @name").reset_index(drop=True)
    else:
        df1 = df.reset_index(drop=True)
    if b6:
        ave_woba = 0.325
        woba = b6_wOBA(df)
        scale = 1.32
    else:
        woba = wOBA(df)
        scale = 1.24
    return round((woba - ave_woba) * seki(df) / scale, 3)


def wRAApPA(
    df, ave_woba=0.33, b6=False
):  # weighted Runs Above Average per Plate Appearance
    if b6:
        ave_woba = 0.325
        woba = b6_wOBA(df)
        scale = 1.32
    else:
        woba = wOBA(df)
        scale = 1.24
    return round((woba - ave_woba) / scale, 3)


def wRC(df, ave_rcppa=0.101, ave_woba=0.33, b6=False):  # weighted Runs Created
    if b6:
        ave_woba = 0.325
        woba = b6_wOBA(df)
        scale = 1.32
    else:
        woba = wOBA(df)
        scale = 1.24
    return round(((woba - ave_woba) / scale + ave_rcppa) * seki(df), 3)


def wRCpPA(
    df, ave_rcppa=0.101, ave_woba=0.33, b6=False
):  # weighted Runs Created per Plate Appearance
    if b6:
        ave_woba = 0.325
        woba = b6_wOBA(df)
        scale = 1.32
    else:
        woba = wOBA(df)
        scale = 1.24
    return round((woba - ave_woba) / scale + ave_rcppa, 3)


def xwOBA(df, compensate=True, only_inplay=False, bb_coef=0.744):
    model = joblib.load(MODEL_PATH)
    df = df.reset_index().rename(columns={"index": "original_index"})
    df_clean = (
        df.query('PitchCall == "InPlay"')
        .dropna(subset=["ExitSpeed", "Angle"])
        .reset_index()
        .rename(columns={"index": "cleaned_index"})
    )

    X = df_clean[["ExitSpeed", "Angle"]]
    df_clean["xwOBAValue"] = model.predict(X)
    df = df.merge(
        df_clean[["original_index", "xwOBAValue"]], on="original_index", how="left"
    )

    ph = ["InPlay", "HitByPitch"]  # 同じく変数です。
    kb = ["Strikeout", "Walk"]  # 同じく変数です。
    bunt = ["Bunt", "GroundBall"]  # バントっていう変数
    df2 = df.query("PitchCall == @ph or KorBB == @kb")
    ds = len(df2)  # 打席数
    bu = len(
        df2.query('PlayResult == "Sacrifice"').query("TaggedHitType == @bunt")
    )  # バント数
    bb = len(df2.query('PitchCall == "HitByPitch" or KorBB == "Walk"'))  # 四死球数

    woba_dict = {
        "Walk": 0.744,
        "Single": 0.938,
        "Double": 1.444,
        "Triple": 2.074,
        "HomeRun": 2.244,
        "Out": 0.0,
        "Sacrifice": 0.0,
    }
    if only_inplay:
        if len(df[~pd.isna(df["xwOBAValue"])]) == 0:
            xwOBA = np.nan
        else:
            xwOBA = round(
                df["xwOBAValue"].sum() / len(df[~pd.isna(df["xwOBAValue"])]), 3
            )
        return xwOBA

    if compensate:
        df["xwOBA"] = df.apply(
            lambda row: (
                woba_dict.get(row["PlayResult"], row["xwOBAValue"])
                if pd.isna(row["xwOBAValue"]) and row["PitchCall"] == "InPlay"
                else row["xwOBAValue"]
            ),
            axis=1,
        )
        if ds - bu != 0:
            xwOBA = round((df["xwOBAValue"].sum() + bb_coef * bb) / (ds - bu), 3)
        else:
            xwOBA = np.nan
    else:
        if ds - bu != 0:
            xwOBA = round(
                (df["xwOBAValue"].sum() + bb_coef * bb)
                / (len(df[~pd.isna(df["xwOBAValue"])]) + bb),
                3,
            )
        else:
            xwOBA = np.nan
    return xwOBA


# モジュールのディレクトリを取得
module_path = os.path.dirname(__file__)
csv_path = os.path.join(module_path, "df_woba_data.csv")

df_woba = pd.read_csv(csv_path)


def calculate_runvalue1(df, sumor100="sum"):
    df = df.dropna(subset=["PlayResult"]).reset_index(drop=True)
    result = 0
    if len(df) == 0:
        return 0
    for i in range(len(df)):
        ball_count = df["Balls"][i]
        strike_count = df["Strikes"][i]
        playresult = df["PlayResult"][i]
        pitchcall = df["PitchCall"][i]
        filtered_rows = df_woba[
            (df_woba["Balls"] == ball_count) & (df_woba["Strikes"] == strike_count)
        ]
        if playresult not in ["Sacrifice", "Undefined"]:
            result += float(filtered_rows[playresult].iloc[0])
        else:
            if pitchcall in ["StrikeCalled", "StrikeSwinging"]:
                pitchcall = "strike"
                result += float(filtered_rows[pitchcall].iloc[0])
            elif pitchcall == "BallCalled":
                pitchcall = "ball"
                result += float(filtered_rows[pitchcall].iloc[0])
            else:
                continue
    if sumor100 == "sum":
        return round(result, 2)
    else:
        return round(result / len(df) * 100, 2)


def calculate_runvalue1_vectorized_final(df, sumor100="sum"):
    """
    最終的に最適化されたベクトル化された calculate_runvalue1 関数。

    Parameters:
    - df (pd.DataFrame): 投球データのデータフレーム。
    - df_woba (pd.DataFrame): wOBA計算用のデータフレーム。
    - sumor100 (str): 'sum' または '100' を指定。

    Returns:
    - float: 計算結果。
    """
    # 'PlayResult' が NaN の行を削除
    df = df.dropna(subset=["PlayResult"]).copy()
    if df.empty:
        return 0.0

    # 'Balls' と 'Strikes' で df と df_woba をマージ
    merged = df.merge(
        df_woba, on=["Balls", "Strikes"], how="left", suffixes=("", "_woba")
    )

    # 'PitchCall' をマッピング
    pitchcall_map = {
        "StrikeCalled": "strike",
        "StrikeSwinging": "strike",
        "BallCalled": "ball",
    }
    merged["PitchCall_mapped"] = merged["PitchCall"].map(pitchcall_map)

    # 'PlayResult' が ['Sacrifice', 'Undefined'] に含まれない場合は 'PlayResult' 列を使用
    condition = ~merged["PlayResult"].isin(["Sacrifice", "Undefined"])
    merged["value_key"] = np.where(
        condition, merged["PlayResult"], merged["PitchCall_mapped"]
    )

    # 有効な value_key を持つ行のみを対象とする
    valid_keys = merged["value_key"].isin(df_woba.columns)
    merged = merged[valid_keys].copy()

    # value_key をインデックスに変換
    # ここでは、df_woba の列名が正しく対応していることを前提とします
    # 例えば、'Single', 'Double', 'strike', 'ball' など
    value_key_df = merged[["value_key"]]

    # 各行の value_num を取得
    # pandas 1.3 以降では、lookup の代わりに DataFrame.melt や他の方法を使用
    # 以下の方法はパフォーマンスが良い
    merged = merged.assign(
        value_num=merged.apply(lambda row: row[row["value_key"]], axis=1)
    )

    # NaN を 0 に置き換える（必要に応じて）
    merged["value_num"] = merged["value_num"].fillna(0.0)

    if sumor100 == "sum":
        result = merged["value_num"].sum()
    else:
        result = (merged["value_num"].mean()) * 100  # 平均をパーセンテージに変換

    return round(result, 2)


# 95マイル=152.9キロ
# ハードヒット率
def Hardp(df, criteria=152.9):
    copy_df = df.copy()
    copy_df.dropna(subset=["ExitSpeed"], inplace=True)
    inplay = len(copy_df.query('PitchCall == "InPlay"'))
    hard = len(copy_df.query("ExitSpeed >= @criteria"))
    bunt = len(copy_df.query('TaggedHitType == "Bunt" and PlayResult == "InPlay"'))
    if inplay - bunt == 0:
        return 0
    else:
        return round(hard / (inplay - bunt) * 100, 1)


# 98マイル=157.7キロ
# バレル率
def Barrelp(df, ev_criteria=157.7, min_ang=26.0, max_ang=30.0):
    copy_df = df.copy()
    copy_df.dropna(subset=["ExitSpeed", "Angle"], inplace=True)
    inplay = len(copy_df.query('PitchCall == "InPlay"'))
    barrel = len(
        copy_df.query("ExitSpeed >= @ev_criteria and @min_ang <= Angle <= @max_ang")
    )
    bunt = len(copy_df.query('TaggedHitType == "Bunt" and PlayResult == "InPlay"'))
    if inplay - bunt == 0:
        return 0
    else:
        return round(barrel / (inplay - bunt) * 100, 1)


# 打席数
def PA(df, name, mode):
    ph = ["InPlay", "HitByPitch"]  # 変数
    kb = ["Strikeout", "Walk"]  # 変数
    if mode == "b":
        df1 = df.query("Batter == @name")
    else:
        df1 = df.query("Pitcher == @name")
    df2 = df1.query("PitchCall == @ph or KorBB == @kb")
    ds = len(df2)  # 打席数
    return ds


def SweetSpot(df=None, name=None):
    if name != None:
        df = df.query("Batter == @name")
    sweetspot_len = len(
        df.query('PitchCall == "InPlay"').query("Angle >= 8 and Angle <= 32")
    )
    all_len = len(df.query('PitchCall == "InPlay"'))
    if all_len == 0:
        sweetspot = np.nan
    else:
        sweetspot = round(sweetspot_len / all_len * 100, 1)
    return sweetspot


def PutAway(df, name=None):
    if name != None:
        df = df.query("Pitcher == @name")
    df1 = df.dropna(subset=["PitchofPA"])
    df2 = df1.query("Strikes == 2")
    df3 = df2.query('KorBB == "Strikeout"')
    putaway = round(len(df3) / len(df2) * 100, 1)
    return putaway


def GB(df, name=None):
    if name != None:
        df = df.query("Pitcher == @name")
    df1 = df.query('PitchCall == "InPlay"')
    dfgoro = df.query('TaggedHitType in ["GroundBall","Bunt"]')
    gb = round(len(dfgoro) / len(df1) * 100, 1)
    return gb


def FB(df, name=None):
    if name != None:
        df = df.query("Pitcher == @name")
    df1 = df.query('PitchCall == "InPlay"')
    dffly = df.query('TaggedHitType in ["FlyBall","PopUp"]')
    fb = round(len(dffly) / len(df1) * 100, 1)
    return fb


def FSS(df, name=None, nx=-0.3, xx=0.3, nz=0.45, xz=1.05):  # First Strike Swing%
    # nx･xxはストライクゾーンの横で、nz･xzはストライクゾーンの縦です。単位はメートル。
    if name != None:
        df1 = df.query("Batter == @name").reset_index()
    else:
        df1 = df.reset_index()
    c1st, m1st = 0, 0
    # mとcはmotherとchildのつもりです。分母と分子。
    plus1st = [
        "StrikeCalled",
        "InPlay",
        "FoulBall",
        "StrikeSwinging",
        "FoulBallFieldable",
        "FoulBallNotFieldable",
    ]
    # 1個ストライク増やすから。まあ変数の名前はどうだっていいんだけどね。
    for index, row in df1.iterrows():
        b = row["Balls"]
        s = row["Strikes"]
        pc = row["PitchCall"]
        px = row["PlateLocSide"]  # Plate X
        pz = row["PlateLocHeight"]  # Plate Z
        if pc in plus1st:
            if s == 0 and nx <= px <= xx and nz <= pz <= xz:
                m1st += 1
                if pc != "StrikeCalled":
                    c1st += 1
    if m1st == 0:
        fss = "-"
    else:
        fss = round(c1st / m1st * 100, 1)
    return fss


def CONP(df, name=None):  # Contact Percent
    if name != None:
        df1 = df.query("Batter == @name").reset_index()
    else:
        df1 = df.reset_index()
    ccon, mcon = 0, 0
    swing = [
        "InPlay",
        "FoulBall",
        "StrikeSwinging",
        "FoulBallFieldable",
        "FoulBallNotFieldable",
    ]
    for index, row in df1.iterrows():
        pc = row["PitchCall"]
        pr = row["PlayResult"]
        if pc in swing:
            mcon += 1
            if pc != "StrikeSwinging":
                ccon += 1
    if mcon == 0:
        conp = "-"
    else:
        conp = round(ccon / mcon * 100, 1)
    return conp
