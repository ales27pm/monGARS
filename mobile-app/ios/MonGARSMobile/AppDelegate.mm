#import "AppDelegate.h"

#import <React/RCTBundleURLProvider.h>

static NSString * const MonGARSAgentTriggerHandoffNotification =
  @"MonGARSAgentTriggerHandoffAvailable";

@implementation AppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
  self.moduleName = @"MonGARSMobile";
  // You can add your custom initial props in the dictionary below.
  // They will be passed down to the ViewController used by React Native.
  self.initialProps = @{};

  BOOL launched = [super application:application didFinishLaunchingWithOptions:launchOptions];
  [UNUserNotificationCenter currentNotificationCenter].delegate = self;
  return launched;
}

- (void)userNotificationCenter:(UNUserNotificationCenter *)center
       willPresentNotification:(UNNotification *)notification
         withCompletionHandler:(void (^)(UNNotificationPresentationOptions options))completionHandler
{
  completionHandler(UNNotificationPresentationOptionBanner |
                    UNNotificationPresentationOptionList |
                    UNNotificationPresentationOptionSound);
}

- (void)userNotificationCenter:(UNUserNotificationCenter *)center
didReceiveNotificationResponse:(UNNotificationResponse *)response
         withCompletionHandler:(void (^)(void))completionHandler
{
  NSDictionary *userInfo = response.notification.request.content.userInfo;
  NSString *triggerID = [userInfo objectForKey:@"monGARSAgentTriggerID"];
  if ([triggerID isKindOfClass:[NSString class]] &&
      [[NSUUID alloc] initWithUUIDString:triggerID] != nil) {
    // Only the opaque trigger identifier and tap time enter UserDefaults. The
    // prompt remains in the package's protected file until a scoped bridge
    // call consumes it once.
    NSDate *tappedAt = [NSDate date];
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    [defaults setObject:triggerID forKey:@"MonGARS.PendingAgentTriggerHandoffID"];
    [defaults setObject:tappedAt forKey:@"MonGARS.PendingAgentTriggerHandoffDate"];

    // A persisted handoff covers cold launch and background resume. This
    // process-local signal also covers a notification tapped while the React
    // Native bridge is already mounted in the foreground.
    [[NSNotificationCenter defaultCenter]
      postNotificationName:MonGARSAgentTriggerHandoffNotification
      object:nil
      userInfo:@{ @"id": triggerID, @"tappedAt": tappedAt }];
  }
  completionHandler();
}

- (NSURL *)sourceURLForBridge:(RCTBridge *)bridge
{
  return [self bundleURL];
}

- (NSURL *)bundleURL
{
#if DEBUG
  return [[RCTBundleURLProvider sharedSettings] jsBundleURLForBundleRoot:@"index"];
#else
  return [[NSBundle mainBundle] URLForResource:@"main" withExtension:@"jsbundle"];
#endif
}

@end
