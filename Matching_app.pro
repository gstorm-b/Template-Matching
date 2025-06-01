QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    matching/blockmax.cpp \
    matching/matchedobject.cpp \
    matching/matcher.cpp \
    matching/matchparams.cpp \
    matching/patternmodel.cpp \
    matching/tmpl.cpp \
    widget/imgroiwidget.cpp \
    widget/roiitem.cpp

HEADERS += \
    mainwindow.h \
    matching/blockmax.h \
    matching/matchedobject.h \
    matching/matcher.h \
    matching/matchparams.h \
    matching/patternmodel.h \
    matching/tmpl.h \
    widget/imgroiwidget.h \
    widget/roiitem.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

win32:CONFIG(release, debug|release): LIBS += -LC:/opencv/build/x64/vc16/lib/ -lopencv_world4110
else:win32:CONFIG(debug, debug|release): LIBS += -LC:/opencv/build/x64/vc16/lib/ -lopencv_world4110
else:unix: LIBS += -LC:/opencv/build/x64/vc16/lib/ -lopencv_world4100

INCLUDEPATH += C:/opencv/build/include
DEPENDPATH += C:/opencv/build/include
