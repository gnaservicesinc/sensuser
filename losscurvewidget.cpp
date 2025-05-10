#include "losscurvewidget.h"
#include <QtWidgets/QVBoxLayout>
#include <QtCharts/QLegend>

LossCurveWidget::LossCurveWidget(QWidget *parent)
    : QWidget(parent),
      m_chartView(new QChartView(this)),
      m_chart(new QChart()),
      m_trainingLossSeries(new QLineSeries(this)),
      m_validationLossSeries(new QLineSeries(this)), // Initialize even if not always used
      m_axisX(new QValueAxis(this)),
      m_axisY(new QValueAxis(this)),
      m_minLoss(0.0), // Or std::numeric_limits<double>::max()
      m_maxLoss(1.0), // Or std::numeric_limits<double>::lowest()
      m_maxEpoch(0)
{
    setupChart();
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->addWidget(m_chartView);
    setLayout(layout);

    // Initial empty state
    clearPlot();
}

LossCurveWidget::~LossCurveWidget() {
    // m_chartView, m_chart, series, and axes are QObjects with this as parent,
    // so they should be deleted automatically. If not, manage manually.
}

void LossCurveWidget::setupChart() {
    m_trainingLossSeries->setName("Training Loss");
    m_validationLossSeries->setName("Validation Loss");

    // Customize series appearance (optional)
    QPen trainingPen(Qt::blue);
    trainingPen.setWidth(2);
    m_trainingLossSeries->setPen(trainingPen);

    QPen validationPen(Qt::green);
    validationPen.setWidth(2);
    m_validationLossSeries->setPen(validationPen);

    m_chart->addSeries(m_trainingLossSeries);
    m_chart->addSeries(m_validationLossSeries); // Add it, visibility can be toggled

    m_chart->setTitle("Training Loss Over Epochs");

    // X Axis (Epochs)
    m_axisX->setTitleText("Epoch");
    m_axisX->setLabelFormat("%d"); // Integer format
    m_axisX->setTickCount(11); // Suggest 10 intervals, adjust as needed
    m_chart->addAxis(m_axisX, Qt::AlignBottom);
    m_trainingLossSeries->attachAxis(m_axisX);
    m_validationLossSeries->attachAxis(m_axisX);

    // Y Axis (Loss)
    m_axisY->setTitleText("Loss");
    m_axisY->setLabelFormat("%.4f"); // Decimal format for loss
    m_axisY->setTickCount(6); // Suggest 5 intervals
    m_chart->addAxis(m_axisY, Qt::AlignLeft);
    m_trainingLossSeries->attachAxis(m_axisY);
    m_validationLossSeries->attachAxis(m_axisY);

    m_chart->legend()->setVisible(true);
    m_chart->legend()->setAlignment(Qt::AlignBottom);

    m_chartView->setChart(m_chart);
    m_chartView->setRenderHint(QPainter::Antialiasing); // For smoother lines
}

void LossCurveWidget::addDataPoint(int epoch, double trainingLoss, double validationLoss) {
    m_trainingLossSeries->append(epoch, trainingLoss);

    if (validationLoss >= 0.0) { // Check if valid validation loss is provided
        m_validationLossSeries->append(epoch, validationLoss);
    }

    // Update axis ranges dynamically
    if (epoch > m_maxEpoch) {
        m_maxEpoch = epoch;
        m_axisX->setMax(m_maxEpoch > 0 ? m_maxEpoch : 1); // Ensure max is at least 1
    }

    bool updateYAxis = false;
    if (trainingLoss < m_minLoss || m_trainingLossSeries->count() == 1) {
        m_minLoss = trainingLoss;
        updateYAxis = true;
    }
    if (trainingLoss > m_maxLoss || m_trainingLossSeries->count() == 1) {
        m_maxLoss = trainingLoss;
        updateYAxis = true;
    }
    if (validationLoss >= 0.0) {
        if (validationLoss < m_minLoss) {
            m_minLoss = validationLoss;
            updateYAxis = true;
        }
        if (validationLoss > m_maxLoss) {
            m_maxLoss = validationLoss;
            updateYAxis = true;
        }
    }

    if (updateYAxis) {
        double range = m_maxLoss - m_minLoss;
        m_axisY->setMin(m_minLoss - range * 0.1); // Add some padding
        m_axisY->setMax(m_maxLoss + range * 0.1); // Add some padding
        if (m_axisY->min() == m_axisY->max()) { // Handle case of single point or flat line
            m_axisY->setMin(m_axisY->min() - 0.5);
            m_axisY->setMax(m_axisY->max() + 0.5);
        }
    }
    m_validationLossSeries->setVisible(m_validationLossSeries->count() > 0); // Show if data exists
}

void LossCurveWidget::updatePlotData(const QVector<QPointF>& trainingData, const QVector<QPointF>& validationData) {
    clearPlot(); // Clear existing data

    m_trainingLossSeries->replace(trainingData);

    if (!validationData.isEmpty()) {
        m_validationLossSeries->replace(validationData);
    }

    // Recalculate axis ranges based on the full dataset
    m_maxEpoch = 0;
    m_minLoss = trainingData.isEmpty() ? 0.0 : trainingData.first().y();
    m_maxLoss = trainingData.isEmpty() ? 1.0 : trainingData.first().y();

    for (const QPointF& point : trainingData) {
        if (point.x() > m_maxEpoch) m_maxEpoch = point.x();
        if (point.y() < m_minLoss) m_minLoss = point.y();
        if (point.y() > m_maxLoss) m_maxLoss = point.y();
    }
    for (const QPointF& point : validationData) {
        if (point.x() > m_maxEpoch) m_maxEpoch = point.x(); // X should align
        if (point.y() < m_minLoss) m_minLoss = point.y();
        if (point.y() > m_maxLoss) m_maxLoss = point.y();
    }

    m_axisX->setMin(0); // Assuming epochs start at 0 or 1
    m_axisX->setMax(m_maxEpoch > 0 ? m_maxEpoch : 1);

    double range = m_maxLoss - m_minLoss;
    if (range < 1e-6) range = 1.0; // Avoid division by zero or tiny range

    m_axisY->setMin(m_minLoss - range * 0.1);
    m_axisY->setMax(m_maxLoss + range * 0.1);
    if (m_axisY->min() >= m_axisY->max()) { // Handle case of single point or flat line
        m_axisY->setMin(m_axisY->min() - 0.5);
        m_axisY->setMax(m_axisY->max() + 0.5);
    }

    m_validationLossSeries->setVisible(!validationData.isEmpty());
    m_chart->update(); // Force redraw
}

void LossCurveWidget::clearPlot() {
    m_trainingLossSeries->clear();
    m_validationLossSeries->clear();
    m_maxEpoch = 0;
    m_minLoss = 0.0; // Or some sensible default min
    m_maxLoss = 1.0; // Or some sensible default max
    m_axisX->setMin(0);
    m_axisX->setMax(1); // Default initial range
    m_axisY->setMin(0.0);
    m_axisY->setMax(1.0); // Default initial range
    m_validationLossSeries->setVisible(false);
}
