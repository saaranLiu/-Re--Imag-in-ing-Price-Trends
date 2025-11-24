import numpy as np
import math
import cv2

from Data import dgp_config as dcf

class DrawChartError(Exception):
    pass

class DrawOHLC(object):
    def __init__(self, df, has_volume_bar=False, ma_lags=None, chart_type="bar"):
        if np.around(df.iloc[0]["Close"], decimals=3) != 1.000:
            raise DrawChartError("Close on first day not equal to 1.")
        self.has_volume_bar = has_volume_bar
        self.vol = df["Vol"] if has_volume_bar else None
        self.ma_lags = ma_lags
        self.ma_name_list = (
            ["ma" + str(ma_lag) for ma_lag in ma_lags] if ma_lags is not None else []
        )
        self.chart_type = chart_type
        assert chart_type in ["bar", "pixel", "centered_pixel"]

        if self.chart_type == "centered_pixel":
            self.df = self.centered_prices(df)
        else:
            self.df = df[["Open", "High", "Low", "Close"] + self.ma_name_list].abs()

        self.ohlc_len = len(df)
        assert self.ohlc_len in [5, 20, 60]
        self.minp = self.df.min().min()
        self.maxp = self.df.max().max()

        (
            self.ohlc_width,
            self.ohlc_height,
            self.volume_height,
        ) = self.__height_and_width()
        self.total_height = self.ohlc_height + self.volume_height + dcf.VOLUME_CHART_GAP if self.has_volume_bar else self.ohlc_height
        first_center = (dcf.BAR_WIDTH - 1) / 2.0
        self.centers = np.arange(
            first_center,
            first_center + dcf.BAR_WIDTH * self.ohlc_len,
            dcf.BAR_WIDTH,
            dtype=int,
        )

    def __height_and_width(self):
        width = dcf.IMAGE_WIDTH[self.ohlc_len]
        total_height = dcf.IMAGE_HEIGHT[self.ohlc_len]
        if self.has_volume_bar:
            volume_height = int(total_height / 5)
            ohlc_height = total_height - volume_height - dcf.VOLUME_CHART_GAP
        else:
            volume_height = 0
            ohlc_height = total_height
        return width, ohlc_height, volume_height

    def __ret_to_yaxis(self, ret):
        pixels_per_unit = (self.ohlc_height - 1.0) / (self.maxp - self.minp)
        res = np.around((ret - self.minp) * pixels_per_unit)
        return int(res)

    def centered_prices(self, df):
        cols = ["Open", "High", "Low", "Close", "Prev_Close"] + self.ma_name_list
        df = df[cols].copy()
        df[cols] = df[cols].div(df["Close"], axis=0)
        df[cols] = df[cols].sub(df["Close"], axis=0)
        df.loc[df.index != 0, self.ma_name_list] = 0
        return df

    def draw_image(self, pattern_list=None):
        if self.maxp == self.minp or math.isnan(self.maxp) or math.isnan(self.minp):
            return None
        try:
            assert (
                self.__ret_to_yaxis(self.minp) == 0
                and self.__ret_to_yaxis(self.maxp) == self.ohlc_height - 1
            )
        except ValueError:
            return None

        # 创建完整图像
        image = np.zeros((self.total_height, self.ohlc_width), dtype=np.uint8)
        
        # 绘制价格部分
        if self.chart_type == "centered_pixel":
            ohlc = self.__draw_centered_pixel_chart()
        else:
            ohlc = self.__draw_ohlc()
            
        # 绘制交易量部分
        if self.vol is not None:
            volume_bar = self.__draw_vol()
            # 将交易量放在底部
            image[:self.volume_height, :] = volume_bar
            # 将价格图表放在上部
            image[self.volume_height + dcf.VOLUME_CHART_GAP:, :] = ohlc
        else:
            image = ohlc

        # 翻转图像以使Y轴从下到上
        image = cv2.flip(image, 0)  # 0表示绕X轴翻转
        return image
    
    def pil_compatible(self, cv_image):
        """将CV2图像转换为与PIL兼容的格式，方便与原代码兼容"""
        from PIL import Image
        import io
        
        # 转换为PIL图像
        pil_img = Image.fromarray(cv_image)
        return pil_img

    def __draw_vol(self):
        volume_bar = np.zeros((self.volume_height, self.ohlc_width), dtype=np.uint8)
        max_volume = np.max(self.vol.abs())
        if (not np.isnan(max_volume)) and max_volume != 0:
            pixels_per_volume = 1.0 * self.volume_height / np.abs(max_volume)
            if not np.around(pixels_per_volume * max_volume) == self.volume_height:
                raise DrawChartError()
                
            for day in range(self.ohlc_len):
                if np.isnan(self.vol.iloc[day]):
                    continue
                vol_height = int(
                    np.around(np.abs(self.vol.iloc[day]) * pixels_per_volume)
                )
                if self.chart_type == "bar":
                    cv2.line(
                        volume_bar,
                        (int(self.centers[day]), 0),
                        (int(self.centers[day]), vol_height - 1),
                        dcf.CHART_COLOR,
                        1
                    )
                elif self.chart_type in ["pixel", "centered_pixel"]:
                    if 0 <= vol_height - 1 < self.volume_height and 0 <= int(self.centers[day]) < self.ohlc_width:
                        volume_bar[vol_height - 1, int(self.centers[day])] = dcf.CHART_COLOR
        
        return volume_bar

    def __draw_ohlc(self):
        ohlc = np.zeros((self.ohlc_height, self.ohlc_width), dtype=np.uint8)
        
        # 绘制移动平均线
        for ma in [self.df[ma_name] for ma_name in self.ma_name_list]:
            for day in range(self.ohlc_len - 1):
                if np.isnan(ma[day]) or np.isnan(ma[day + 1]):
                    continue
                if self.chart_type == "bar":
                    cv2.line(
                        ohlc,
                        (int(self.centers[day]), self.__ret_to_yaxis(ma[day])),
                        (int(self.centers[day + 1]), self.__ret_to_yaxis(ma[day + 1])),
                        dcf.CHART_COLOR,
                        1
                    )
                elif self.chart_type == "pixel":
                    if 0 <= self.__ret_to_yaxis(ma[day]) < self.ohlc_height and 0 <= int(self.centers[day]) < self.ohlc_width:
                        ohlc[self.__ret_to_yaxis(ma[day]), int(self.centers[day])] = dcf.CHART_COLOR
            
            try:
                if 0 <= self.__ret_to_yaxis(ma[self.ohlc_len - 1]) < self.ohlc_height and 0 <= int(self.centers[self.ohlc_len - 1]) < self.ohlc_width:
                    ohlc[self.__ret_to_yaxis(ma[self.ohlc_len - 1]), int(self.centers[self.ohlc_len - 1])] = dcf.CHART_COLOR
            except (ValueError, IndexError):
                pass

        # 绘制OHLC柱
        for day in range(self.ohlc_len):
            highp_today = self.df["High"].iloc[day]
            lowp_today = self.df["Low"].iloc[day]
            closep_today = self.df["Close"].iloc[day]
            openp_today = self.df["Open"].iloc[day]

            if np.isnan(highp_today) or np.isnan(lowp_today):
                continue
                
            left = int(math.ceil(self.centers[day] - int(dcf.BAR_WIDTH / 2)))
            right = int(math.floor(self.centers[day] + int(dcf.BAR_WIDTH / 2)))

            line_left = int(math.ceil(self.centers[day] - int(dcf.LINE_WIDTH / 2)))
            line_right = int(math.floor(self.centers[day] + int(dcf.LINE_WIDTH / 2)))

            line_bottom = self.__ret_to_yaxis(lowp_today)
            line_up = self.__ret_to_yaxis(highp_today)

            if self.chart_type == "bar":
                cv2.rectangle(
                    ohlc,
                    (line_left, line_bottom),
                    (line_right, line_up),
                    dcf.CHART_COLOR,
                    -1  # 填充矩形
                )
            elif self.chart_type == "pixel":
                if 0 <= line_bottom < self.ohlc_height and 0 <= int(self.centers[day]) < self.ohlc_width:
                    ohlc[line_bottom, int(self.centers[day])] = dcf.CHART_COLOR
                if 0 <= line_up < self.ohlc_height and 0 <= int(self.centers[day]) < self.ohlc_width:
                    ohlc[line_up, int(self.centers[day])] = dcf.CHART_COLOR
            
            # 绘制开盘价
            if not np.isnan(openp_today):
                open_line = self.__ret_to_yaxis(openp_today)
                for i in range(left, int(self.centers[day]) + 1):
                    if 0 <= open_line < self.ohlc_height and 0 <= i < self.ohlc_width:
                        ohlc[open_line, i] = dcf.CHART_COLOR

            # 绘制收盘价
            if not np.isnan(closep_today):
                close_line = self.__ret_to_yaxis(closep_today)
                for i in range(int(self.centers[day]) + 1, right + 1):
                    if 0 <= close_line < self.ohlc_height and 0 <= i < self.ohlc_width:
                        ohlc[close_line, i] = dcf.CHART_COLOR

        return ohlc

    def __draw_centered_pixel_chart(self):
        ohlc = np.zeros((self.ohlc_height, self.ohlc_width), dtype=np.uint8)
        
        for day in range(self.ohlc_len):
            highp_today = self.df["High"].iloc[day]
            lowp_today = self.df["Low"].iloc[day]
            prev_closep_today = self.df["Prev_Close"].iloc[day]
            openp_today = self.df["Open"].iloc[day]

            if np.isnan(highp_today) or np.isnan(lowp_today):
                continue

            high_y = self.__ret_to_yaxis(highp_today)
            low_y = self.__ret_to_yaxis(lowp_today)
            
            if 0 <= high_y < self.ohlc_height and 0 <= int(self.centers[day]) < self.ohlc_width:
                ohlc[high_y, int(self.centers[day])] = dcf.CHART_COLOR
            if 0 <= low_y < self.ohlc_height and 0 <= int(self.centers[day]) < self.ohlc_width:
                ohlc[low_y, int(self.centers[day])] = dcf.CHART_COLOR

            left = int(math.ceil(self.centers[day] - int(dcf.BAR_WIDTH / 2)))
            right = int(math.floor(self.centers[day] + int(dcf.BAR_WIDTH / 2)))

            if not np.isnan(openp_today):
                open_line = self.__ret_to_yaxis(openp_today)
                for i in range(left, int(self.centers[day]) + 1):
                    if 0 <= open_line < self.ohlc_height and 0 <= i < self.ohlc_width:
                        ohlc[open_line, i] = dcf.CHART_COLOR

            if not np.isnan(prev_closep_today):
                prev_close_line = self.__ret_to_yaxis(prev_closep_today)
                for i in range(left, right + 1):
                    if 0 <= prev_close_line < self.ohlc_height and 0 <= i < self.ohlc_width:
                        ohlc[prev_close_line, i] = dcf.CHART_COLOR

        # 绘制移动平均线
        for ma in [self.df[ma_name] for ma_name in self.ma_name_list]:
            day = 0
            if not np.isnan(ma[day]):
                ma_y = self.__ret_to_yaxis(ma[day])
                if 0 <= ma_y < self.ohlc_height:
                    cv2.line(ohlc, (0, ma_y), (3, ma_y), dcf.CHART_COLOR, 1)

        return ohlc