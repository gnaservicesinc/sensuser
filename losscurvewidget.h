#ifndef LOSSCURVEWIDGET_H
#define LOSSCURVEWIDGET_H

#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QtWidgets/QWidget>

class LossCurveWidget : public QWidget {
    Q_OBJECT
public:
    explicit LossCurveWidget(QWidget *parent = nullptr);
    ~LossCurveWidget();

public slots:
    void addDataPoint(int epoch, double trainingLoss, double validationLoss = -1.0); // validationLoss is optional
    void updatePlotData(const QVector<QPointF>& trainingData, const QVector<QPointF>& validationData = QVector<QPointF>());
    void clearPlot(); // To reset for a new training session

private:
    void setupChart();

    QChartView *m_chartView;
    QChart *m_chart;
    QLineSeries *m_trainingLossSeries;
    QLineSeries *m_validationLossSeries; // Optional: for validation loss
    QValueAxis *m_axisX;
    QValueAxis *m_axisY;

    double m_minLoss;
    double m_maxLoss;
    int m_maxEpoch;
};

#endif // LOSSCURVEWIDGET_H
