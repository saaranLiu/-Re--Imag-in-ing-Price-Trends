from __future__ import print_function, division
import numpy as np
import math
from PIL import Image, ImageDraw

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
        first_center = (dcf.BAR_WIDTH - 1) / 2.0
        self.centers = np.arange(
            first_center,
            first_center + dcf.BAR_WIDTH * self.ohlc_len,
            dcf.BAR_WIDTH,
            dtype=int,
        )

    def __height_and_width(self):
        width, height = dcf.IMAGE_WIDTH[self.ohlc_len], dcf.IMAGE_HEIGHT[self.ohlc_len]
        if self.has_volume_bar:
            volume_height = int(height / 5)
            price_height = height - volume_height - dcf.VOLUME_CHART_GAP
        else:
            volume_height = 0
            price_height = height
        return width, price_height, volume_height

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

        if self.chart_type == "centered_pixel":
            ohlc = self.__draw_centered_pixel_chart()
        else:
            ohlc = self.__draw_ohlc()

        if self.vol is not None:
            volume_bar = self.__draw_vol()
            image = Image.new(
                "L",
                (
                    self.ohlc_width,
                    self.ohlc_height + self.volume_height + dcf.VOLUME_CHART_GAP,
                ),
            )
            image.paste(ohlc, (0, self.volume_height + dcf.VOLUME_CHART_GAP))
            image.paste(volume_bar, (0, 0))
        else:
            image = ohlc

        if pattern_list is not None:
            cur_day = 0
            draw = ImageDraw.Draw(image)
            for pat, length in pattern_list:
                if pat is None:
                    cur_day += length
                else:
                    draw.line(
                        [
                            self.centers[cur_day],
                            0,
                            self.centers[cur_day],
                            self.ohlc_height - 1,
                        ],
                        fill=dcf.CHART_COLOR,
                    )
                    cur_day += length
                    if cur_day < self.ohlc_len:
                        draw.line(
                            [
                                self.centers[cur_day],
                                0,
                                self.centers[cur_day],
                                self.ohlc_height - 1,
                            ],
                            fill=dcf.CHART_COLOR,
                        )
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def __draw_vol(self):
        volume_bar = Image.new(
            "L", (self.ohlc_width, self.volume_height), dcf.BACKGROUND_COLOR
        )
        pixels = volume_bar.load()
        max_volume = np.max(self.vol.abs())
        if (not np.isnan(max_volume)) and max_volume != 0:
            pixels_per_volume = 1.0 * self.volume_height / np.abs(max_volume)
            if not np.around(pixels_per_volume * max_volume) == self.volume_height:
                raise DrawChartError()
            draw = ImageDraw.Draw(volume_bar)
            for day in range(self.ohlc_len):
                if np.isnan(self.vol.iloc[day]):
                    continue
                vol_height = int(
                    np.around(np.abs(self.vol.iloc[day]) * pixels_per_volume)
                )
                if self.chart_type == "bar":
                    draw.line(
                        [self.centers[day], 0, self.centers[day], vol_height - 1],
                        fill=dcf.CHART_COLOR,
                    )
                elif self.chart_type in ["pixel", "centered_pixel"]:
                    pixels[int(self.centers[day]), vol_height - 1] = dcf.CHART_COLOR
                else:
                    raise ValueError(f"Chart type {self.chart_type} not supported")
            del draw
        return volume_bar

    def __draw_ohlc(self):
        ohlc = Image.new(
            "L", (self.ohlc_width, self.ohlc_height), dcf.BACKGROUND_COLOR
        )
        pixels = ohlc.load()
        for ma in [self.df[ma_name] for ma_name in self.ma_name_list]:
            draw = ImageDraw.Draw(ohlc)
            for day in range(self.ohlc_len - 1):
                if np.isnan(ma[day]) or np.isnan(ma[day + 1]):
                    continue
                if self.chart_type == "bar":
                    draw.line(
                        (
                            self.centers[day],
                            self.__ret_to_yaxis(ma[day]),
                            self.centers[day + 1],
                            self.__ret_to_yaxis(ma[day + 1]),
                        ),
                        width=1,
                        fill=dcf.CHART_COLOR,
                    )
                elif self.chart_type == "pixel":
                    pixels[
                        int(self.centers[day]), self.__ret_to_yaxis(ma[day])
                    ] = dcf.CHART_COLOR
                else:
                    raise ValueError(f"Chart type {self.chart_type} not supported")
            try:
                pixels[
                    int(self.centers[self.ohlc_len - 1]),
                    self.__ret_to_yaxis(ma[self.ohlc_len - 1]),
                ] = dcf.CHART_COLOR
            except ValueError:
                pass
            del draw

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
                for i in range(line_left, line_right + 1):
                    for j in range(line_bottom, line_up + 1):
                        pixels[i, j] = dcf.CHART_COLOR
            elif self.chart_type == "pixel":
                pixels[int(self.centers[day]), line_bottom] = dcf.CHART_COLOR
                pixels[int(self.centers[day]), line_up] = dcf.CHART_COLOR
            else:
                raise ValueError(f"Chart type {self.chart_type} not supported")

            if not np.isnan(openp_today):
                open_line = self.__ret_to_yaxis(openp_today)
                for i in range(left, int(self.centers[day]) + 1):
                    j = open_line
                    pixels[i, j] = dcf.CHART_COLOR

            if not np.isnan(closep_today):
                close_line = self.__ret_to_yaxis(closep_today)
                for i in range(int(self.centers[day]) + 1, right + 1):
                    j = close_line
                    pixels[i, j] = dcf.CHART_COLOR

        return ohlc

    def __draw_centered_pixel_chart(self):
        ohlc = Image.new(
            "L", (self.ohlc_width, self.ohlc_height), dcf.BACKGROUND_COLOR
        )
        pixels = ohlc.load()

        for day in range(self.ohlc_len):
            highp_today = self.df["High"].iloc[day]
            lowp_today = self.df["Low"].iloc[day]
            prev_closep_today = self.df["Prev_Close"].iloc[day]
            openp_today = self.df["Open"].iloc[day]

            if np.isnan(highp_today) or np.isnan(lowp_today):
                continue

            pixels[
                int(self.centers[day]), self.__ret_to_yaxis(highp_today)
            ] = dcf.CHART_COLOR
            pixels[
                int(self.centers[day]), self.__ret_to_yaxis(lowp_today)
            ] = dcf.CHART_COLOR

            left = int(math.ceil(self.centers[day] - int(dcf.BAR_WIDTH / 2)))
            right = int(math.floor(self.centers[day] + int(dcf.BAR_WIDTH / 2)))

            if not np.isnan(openp_today):
                open_line = self.__ret_to_yaxis(openp_today)
                for i in range(left, int(self.centers[day]) + 1):
                    j = open_line
                    pixels[i, j] = dcf.CHART_COLOR

            if not np.isnan(prev_closep_today):
                prev_close_line = self.__ret_to_yaxis(prev_closep_today)
                for i in range(left, right + 1):
                    j = prev_close_line
                    pixels[i, j] = dcf.CHART_COLOR

        for ma in [self.df[ma_name] for ma_name in self.ma_name_list]:
            draw = ImageDraw.Draw(ohlc)
            day = 0
            if np.isnan(ma[day]) or np.isnan(ma[day + 1]):
                continue
            draw.line(
                (0, self.__ret_to_yaxis(ma[day]), 3, self.__ret_to_yaxis(ma[day])),
                width=1,
                fill=dcf.CHART_COLOR,
            )
            del draw

        return ohlc
